# =============================================================================
#  steps_bc5cdr.py   [BC5CDR-Chem]   code version 2.1
#  Step-matched evaluation of data augmentation for few-shot BioNER
#  (revision version; replaces steps_ncbi.py, steps_bc5cdr.py, steps_bc2gm.py)
#
#  Corpus-specific copy for BC5CDR-Chem. The three copies (steps_ncbi.py,
#  steps_bc5cdr.py, steps_bc2gm.py) are identical except for CORPUS, so they
#  can run in parallel on separate Kaggle accounts.
#
#  Design principle: every run is defined by an explicit INTEGER number of
#  optimizer steps (max_steps). The epoch count is derived from it and logged
#  as an effective (possibly fractional) value; it is never the other way round.
#
#  Budgets (per few-shot cell and seed, with B = batch size, n = |D_r|):
#    S_b5   = ceil(n / B) * 5                 baseline reference (5 epochs)
#    S_a5   = ceil((n + n_elig*K) / B) * 5    augmented reference (5 epochs)
#    'kx'   = k * S_b5                        common step grid (k = 1,2,8,16)
#    'match'= S_a5                            step-matched budget
#  Because every method is run at the same integer budgets, curves of Base,
#  MR and COSINER are compared point by point at identical step counts.
#
#  Phases (executed in this order, so an interrupted session still leaves the
#  primary step-matched comparisons complete for all cells):
#    1  primary configurations: Base@1x, Base@match, Dup@match, MR@match,
#       COSINER@match                                     (all 9 cells)
#    2  training-budget sweep for Base, MR and COSINER    (all 9 cells)
#    3  repeat runs to quantify run-to-run variability
#
#  Outputs, in OUT_DIR/steps_<slug>/ :
#    results.csv                one row per run
#    counts_<cell>_s<seed>.npz  per-test-sentence TP/FP/FN (overall, seen,
#                               unseen) for every run; needed for the two-way
#                               (seed x test-sentence) cluster bootstrap
#    aug_<cell>_s<seed>.json.gz few-shot subset IDs, dev IDs and the MR and
#                               COSINER augmented sets (cache + data release)
#    aug_stats.csv              replacement statistics per cell/seed/method
#    meta.json                  software, hardware, data hashes, settings
#
#  Resumable: add the output of a previous session as notebook Input; the
#  script merges it and skips finished runs.
# =============================================================================

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import re, csv, json, gzip, time, math, random, hashlib, platform, shutil
import socket, warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (AutoTokenizer, AutoModel, AutoConfig,
                          AutoModelForTokenClassification,
                          DataCollatorForTokenClassification)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
try:
    hf_logging.disable_progress_bar()
except Exception:
    pass
warnings.filterwarnings('ignore')


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None else cast(v)


# =============================================================================
# USER SETTINGS
# =============================================================================
CORPUS        = 'BC5CDR-Chem'   # fixed for this file
SEED_SHARD    = None       # e.g. [64, 128, 256, 512, 1024] to split seeds over sessions
PHASES        = (1, 2, 3)  # see header
CELLS_TO_RUN  = ['2%']     # few-shot cells for THIS session: ['2%'], then ['5%'], then ['10%']
                           # (or ['2%', '5%', '10%'] for all). Subsets do not depend
                           # on this choice, so splitting cells over sessions is safe.
DRY_RUN       = _env('STEPS_DRY_RUN', '0') == '1'  # True: print the run plan and cost only
USE_AMP       = False      # fp16 mixed precision. Faster on T4, but may skip a few
                           # optimizer steps (logged as n_skipped). Keep False for
                           # strict step parity.
EXPORT_AUG    = True       # store augmented sets (cache + data release)
TIME_BUDGET_H = 11.3       # stop starting new runs after this many hours (Kaggle: 12 h)

INPUT_ROOT = _env('STEPS_INPUT_ROOT', '/kaggle/input')
OUT_DIR    = _env('STEPS_OUT_DIR', '/kaggle/working')
DATA_ROOT  = _env('STEPS_DATA_ROOT', '/kaggle/input/datasets/wantodbz')

# =============================================================================
# EXPERIMENTAL PROTOCOL (fixed; changing these changes the study)
# =============================================================================
MODEL_NAME      = _env('STEPS_MODEL', 'dmis-lab/biobert-v1.1')
MODEL_REVISION  = None     # resolved commit hash is logged in meta.json
LEARNING_RATE   = 5e-5
WEIGHT_DECAY    = 0.01     # torch.optim.AdamW default, stated explicitly
ADAM_BETAS      = (0.9, 0.999)
ADAM_EPS        = 1e-8
GRAD_CLIP       = 1.0
BATCH_SIZE      = 8
EVAL_BATCH_SIZE = 64
MAX_SEQ_LEN     = 512
REF_EPOCHS      = 5        # reference protocol of the literature

SEEDS = [int(s) for s in _env('STEPS_SEEDS',
         '64,128,256,512,1024,2048,4096,8192,16384,32768').split(',')]
AUG_K = 5                  # copies per mention-bearing sentence (Dup, MR, COSINER)
RHO   = 3                  # at most RHO*K candidate ranks are tried per sentence
FRACTIONS = [(0.02, '2%'), (0.05, '5%'), (0.10, '10%')]

PRIMARY_RUNS  = [('Base', '1x'), ('Base', 'match'), ('Dup', 'match'),
                 ('MR', 'match'), ('COSINER', 'match')]
SWEEP_METHODS = ['Base', 'MR', 'COSINER']
# Budget sweep only in the 2% cells, i.e. exactly where the baseline epoch sweep
# of the first submission was run (Reviewer 1, comment 1: comparable sweeps for
# both approaches). The 5% and 10% cells keep the step-matched comparison only,
# so phase 2 is empty there.
SWEEP_BUDGETS = {'2%':  ['1x', '2x', 'match', '8x', '16x'],
                 '5%':  [],
                 '10%': []}
REPEAT_RUNS   = [('Base', 'match'), ('MR', 'match')]
REPEAT_CELLS  = ['2%']
REPEAT_SEEDS  = SEEDS[:3]

# Names used in the first submission, kept for traceability.
LEGACY = {'Base@1x': 'Base-E5', 'Base@2x': 'Base-E10', 'Base@4x': 'Base-E20',
          'Base@8x': 'Base-E40', 'Base@16x': 'Base-E80', 'Base@match': 'Base-match',
          'Dup@match': 'Duplicate', 'MR@match': 'MR', 'COSINER@match': 'COSINER'}

CORPORA = {
    'NCBI-Disease': dict(kind='generic', entity='Disease', slug='ncbi', dir='ncbi-disease'),
    'BC5CDR-Chem':  dict(kind='bc5cdr',  entity='Chemical', slug='bc5cdr', dir='bc5cdr'),
    'BC2GM':        dict(kind='generic', entity='Gene', slug='bc2gm', dir='bc2gm-corpus'),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ON = USE_AMP and device == 'cuda'

# v2.1: 'eager' attention. The default SDPA kernel on a T4 (memory-efficient
# attention) has a non-deterministic backward pass that use_deterministic_
# algorithms(warn_only=True) does not remove; it made one of six identical
# repeat runs in v2.0 differ by 0.94 F1. Results of v2.0 are not reused.
CODE_VERSION = '2.1'
ATTN_IMPL = 'eager'
STRICT_DETERMINISM = True   # abort if the start-up determinism check fails


# =============================================================================
# REPRODUCIBILITY HELPERS
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def md5_obj(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=False).encode()).hexdigest()


def cell_tag(label):
    return {'2%': '02pct', '5%': '05pct', '10%': '10pct'}[label]


# =============================================================================
# DATA LOADING
# =============================================================================
def parse_token_list(cell):
    """Parse a token-list cell in Python-list style ("['a', 'b']") or numpy
    repr style ("['a' 'b']"), including tokens containing apostrophes, which
    are written with double quotes (e.g. "5'-nucleotidase")."""
    t = str(cell).strip()
    if t.startswith('[') and t.endswith(']'):
        t = t[1:-1]
    out, i, n = [], 0, len(t)
    while i < n:
        if t[i] in '"\'':
            q = t[i]; i += 1; buf = []
            while i < n:
                if t[i] == '\\' and i + 1 < n:
                    buf.append(t[i + 1]); i += 2; continue
                if t[i] == q and (i + 1 >= n or t[i + 1] in ' ,\t\n'):
                    break
                buf.append(t[i]); i += 1
            out.append(''.join(buf)); i += 1
        else:
            i += 1
    return out


def parse_tag_list(cell):
    return [int(x) for x in re.findall(r'-?\d+', str(cell))]


def parse_file(path, kind, entity, split):
    """Returns (sentences, report). Sentence IDs are '<split>:<row id>'.
    The same token parser is used for every corpus (the first submission used
    a weaker regex for BC5CDR that silently dropped apostrophe tokens)."""
    if kind == 'bc5cdr':
        tag_map = {0: 'O', 1: 'B-' + entity, 2: 'O', 3: 'O', 4: 'I-' + entity}
    else:
        tag_map = {0: 'O', 1: 'B-' + entity, 2: 'I-' + entity}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    tok_col = 'tokens' if 'tokens' in df.columns else df.columns[0]
    tag_col = 'ner_tags' if 'ner_tags' in df.columns else df.columns[1]
    out, dropped, unknown = [], [], Counter()
    ids = df['id'].tolist() if 'id' in df.columns else [''] * len(df)
    for ri, (tcell, gcell, rid) in enumerate(zip(df[tok_col].tolist(),
                                                 df[tag_col].tolist(), ids)):
        tokens = parse_token_list(tcell)
        raw = parse_tag_list(gcell)
        for t in raw:
            if t not in tag_map:
                unknown[t] += 1
        tags = [tag_map.get(t, 'O') for t in raw]
        rid = rid if str(rid) != '' else ri
        if tokens and len(tokens) == len(tags):
            out.append({'id': f'{split}:{rid}', 'tokens': tokens, 'tags': tags})
        else:
            dropped.append(dict(row=ri, id=str(rid), n_tok=len(tokens), n_tag=len(tags)))
    rep = dict(file=os.path.basename(path), md5=md5_file(path), n_rows=len(df),
               n_kept=len(out), n_dropped=len(dropped), dropped=dropped,
               unknown_tag_values=dict(unknown))
    if dropped:
        print(f'    [parser] {split}: dropped {len(dropped)}/{len(df)} rows '
              f'(token/tag length mismatch) -> listed in meta.json')
    return out, rep


def resolve_path(path, corpus, split):
    if os.path.exists(path):
        return path
    slug = os.path.basename(os.path.dirname(path)); fname = os.path.basename(path)
    cands = []
    for root, _d, files in os.walk(INPUT_ROOT):
        for f in files:
            if not f.endswith('.csv'):
                continue
            full = os.path.join(root, f); score = 0
            if slug in full: score += 2
            if f == fname: score += 2
            elif split in f.lower(): score += 1
            if score >= 3: cands.append((score, full))
    if cands:
        cands.sort(reverse=True)
        print(f'    [path] {corpus}/{split}: using {cands[0][1]}')
        return cands[0][1]
    raise FileNotFoundError(f'{corpus}/{split}: not found ({path}).')


def load_corpus(corpus):
    spec = CORPORA[corpus]; d = os.path.join(DATA_ROOT, spec['dir'])
    data, reps = {}, {}
    for split, fname in [('train', 'train.csv'), ('validation', 'validation.csv'),
                         ('test', 'test.csv')]:
        p = resolve_path(os.path.join(d, fname), corpus, split)
        data[split], reps[split] = parse_file(p, spec['kind'], spec['entity'], split)
        reps[split]['path'] = p
    return data, reps


def shuffled_prefix(n_total, k, seed):
    """First k indices of a seeded permutation. Using the same permutation for
    every fraction makes the 2% subset a subset of the 5% subset, which is a
    subset of the 10% subset (nested design, handled in the analysis)."""
    rng = random.Random(seed)
    idx = list(range(n_total)); rng.shuffle(idx)
    return idx[:k]


def few_shot_indices(n_total, frac, seed):
    return shuffled_prefix(n_total, max(1, int(n_total * frac)), seed)


def dev_indices(n_val, n_dev, seed):
    """Few-shot development set drawn from the validation split, same size as
    the few-shot training subset. Used ONLY to select a training budget in the
    sweep analysis; never for checkpoint selection inside a run."""
    return shuffled_prefix(n_val, min(n_dev, n_val), 1_000_003 + seed)


# =============================================================================
# SPANS, LEXICON, RANKINGS, REPLACEMENT OPERATOR
# =============================================================================
def iob_spans(tags):
    """(start, end, type) spans, conlleval behaviour: an I-X after O or after a
    different type opens a new chunk (identical to seqeval default mode)."""
    spans, i = [], 0
    while i < len(tags):
        t = tags[i]
        if t == 'O':
            i += 1; continue
        typ = t[2:]; st = i; i += 1
        while i < len(tags) and tags[i].startswith('I-') and tags[i][2:] == typ:
            i += 1
        spans.append((st, i, typ))
    return spans


def mentions_of(sent):
    tok = sent['tokens']
    return [(st, en, ' '.join(tok[st:en]), et) for (st, en, et) in iob_spans(sent['tags'])]


def extract_mentions(sentences):
    contexts = defaultdict(list); types = {}
    for s in sentences:
        for (st, en, mtext, et) in mentions_of(s):
            contexts[mtext].append(' '.join(s['tokens'])); types[mtext] = et
    return contexts, types


@torch.no_grad()
def encode_sentences(texts, tok, enc, batch=32):
    """Mean of last hidden states, excluding [CLS] and [SEP]."""
    out = {}; texts = sorted(set(texts))
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        b = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                max_length=MAX_SEQ_LEN)
        b = {k: v.to(device) for k, v in b.items()}
        h = enc(**b).last_hidden_state.float()
        mask = b['attention_mask'].clone(); mask[:, 0] = 0
        for j in range(mask.size(0)):
            mask[j, int(b['attention_mask'][j].sum().item()) - 1] = 0
        m = mask.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        for j, t in enumerate(chunk):
            out[t] = v[j].cpu().numpy().astype(np.float64)
    return out


def build_rankings(subset, tok, enc):
    """COSINER protocol: contextual concept embeddings and cosine ranking of
    same-type candidate mentions from the few-shot lexicon."""
    contexts, types = extract_mentions(subset)
    sent_vec = encode_sentences([s for c in contexts.values() for s in c], tok, enc)
    emb = {}
    for m, ctxs in contexts.items():
        lr = 1.0 / len(ctxs); v = None
        for s in ctxs:
            vc = sent_vec[s]
            if v is None:
                v = vc.copy()
            else:
                den = np.linalg.norm(v) * np.linalg.norm(vc) + 1e-8
                sim = max(0.0, float(np.dot(v, vc) / den))
                v = v + lr * (1 - sim) * vc
        emb[m] = v
    names = list(emb.keys())
    if len(names) < 2:
        return {}
    M = np.array([emb[m] for m in names])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sim = M @ M.T
    rankings = {}
    for i, m in enumerate(names):
        mt = types[m]
        sc = [(names[j], float(sim[i, j])) for j in range(len(names))
              if i != j and types[names[j]] == mt]
        sc.sort(key=lambda x: (-x[1], x[0]))      # deterministic tie-break
        rankings[m] = sc
    return rankings


def shuffled_rankings(rankings, seed):
    """MR: same operator, candidate set and budget as COSINER; random order."""
    rng = random.Random(seed); out = {}
    for m in sorted(rankings):
        c = list(rankings[m]); rng.shuffle(c); out[m] = c
    return out


def make_copy(sent, rankings, aug_idx, tag):
    """Copy that replaces every mention by its aug_idx-th ranked candidate.
    The token list is rebuilt from scratch, so a longer replacement cannot
    overwrite neighbouring tokens. Returns None if nothing was replaced."""
    tok, tg = sent['tokens'], sent['tags']
    ms = mentions_of(sent)
    new_tok, new_tg = [], []
    cursor, n_rep, d_len = 0, 0, 0
    for (st, en, mtext, et) in ms:
        new_tok.extend(tok[cursor:st]); new_tg.extend(tg[cursor:st])
        sims = rankings.get(mtext)
        if sims is not None and aug_idx < len(sims):
            rep = sims[aug_idx][0].split()
            new_tok.extend(rep)
            new_tg.extend(['B-' + et] + ['I-' + et] * (len(rep) - 1))
            d_len += abs(len(rep) - (en - st)); n_rep += 1
        else:
            new_tok.extend(tok[st:en]); new_tg.extend(tg[st:en])
        cursor = en
    new_tok.extend(tok[cursor:]); new_tg.extend(tg[cursor:])
    if n_rep == 0:
        return None
    assert len(new_tok) == len(new_tg)
    return ({'id': f"{sent['id']}#{tag}{aug_idx}", 'tokens': new_tok, 'tags': new_tg},
            n_rep, len(ms), d_len)


def augment(eligible, rankings, tag):
    """First K successful copies per sentence, trying ranks 0..RHO*K-1."""
    aug = []
    st = Counter()
    for s in eligible:
        made = 0
        for a in range(RHO * AUG_K):
            if made == AUG_K:
                break
            st['attempts'] += 1
            r = make_copy(s, rankings, a, tag)
            if r is None:
                st['failed_attempts'] += 1; continue
            cp, n_rep, n_m, d_len = r
            aug.append(cp); made += 1
            st['replacements'] += n_rep; st['mentions_in_copies'] += n_m
            st['abs_dlen'] += d_len
        if made < AUG_K:
            st['sent_short'] += 1
    n = max(1, len(aug))
    stats = dict(copies_target=len(eligible) * AUG_K, copies_made=len(aug),
                 attempts=st['attempts'], failed_attempts=st['failed_attempts'],
                 sent_short=st['sent_short'],
                 rep_per_copy=st['replacements'] / n,
                 frac_mentions_replaced=st['replacements'] / max(1, st['mentions_in_copies']),
                 mean_abs_dlen=st['abs_dlen'] / max(1, st['replacements']))
    return aug, stats


def dup_set(eligible):
    return [{'id': f"{s['id']}#dup{k}", 'tokens': list(s['tokens']), 'tags': list(s['tags'])}
            for s in eligible for k in range(AUG_K)]


# =============================================================================
# DATASET, TRAINING, EVALUATION
# =============================================================================
class NERDataset(Dataset):
    """Unpadded sequences; the collator pads per batch.
    Training: continuation sub-tokens get the I- label of their word.
    Evaluation: only the first sub-token of each word is labelled; word_ids
    are kept so predictions are mapped back to word positions exactly."""

    def __init__(self, sentences, tokenizer, tag2idx, eval_mode=False):
        self.s, self.t, self.m, self.eval_mode = sentences, tokenizer, tag2idx, eval_mode
        self.cache = [self._encode(i) for i in range(len(sentences))] if eval_mode else None

    def __len__(self):
        return len(self.s)

    def _encode(self, i):
        tokens, tags = self.s[i]['tokens'], self.s[i]['tags']
        e = self.t(tokens, is_split_into_words=True, max_length=MAX_SEQ_LEN, truncation=True)
        wid = e.word_ids(); lab = []; prev = None; first = []
        for j, w in enumerate(wid):
            if w is None:
                lab.append(-100)
            elif w != prev:
                lab.append(self.m[tags[w]]); first.append((j, w))
            elif self.eval_mode:
                lab.append(-100)
            else:
                tg = tags[w]
                lab.append(self.m['I-' + tg[2:]] if tg.startswith('B-') else self.m[tg])
            prev = w
        return {'input_ids': e['input_ids'], 'attention_mask': e['attention_mask'],
                'labels': lab}, first

    def __getitem__(self, i):
        return (self.cache[i] if self.eval_mode else self._encode(i))[0]

    def first_positions(self, i):
        return self.cache[i][1]


def linear_decay(opt, total):
    """LR decays linearly from LEARNING_RATE to 0 over exactly `total` steps,
    no warm-up (identical to transformers' linear schedule with 0 warm-up)."""
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, (total - s) / max(1, total)))


def make_scaler():
    try:
        return torch.amp.GradScaler('cuda', enabled=AMP_ON)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=AMP_ON)


def train(train_s, seed, max_steps, tokenizer, tag2idx, collator):
    set_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(tag2idx), revision=MODEL_REVISION,
        attn_implementation=ATTN_IMPL).to(device)
    g = torch.Generator(); g.manual_seed(seed)        # data order independent of init
    loader = DataLoader(NERDataset(train_s, tokenizer, tag2idx), batch_size=BATCH_SIZE,
                        shuffle=True, generator=g, collate_fn=collator, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
                            eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    sch = linear_decay(opt, max_steps)
    scaler = make_scaler()
    step, skipped = 0, 0
    model.train()
    while step < max_steps:
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP_ON):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            before = scaler.get_scale() if AMP_ON else None
            scaler.step(opt); scaler.update()
            if AMP_ON and scaler.get_scale() < before:
                skipped += 1
            sch.step(); step += 1
            if step >= max_steps:
                break
    return model, step, skipped, len(loader)


@torch.no_grad()
def predict(model, ds, collator, idx2tag):
    """Word-level predictions for every sentence. Words without a labelled
    sub-token (truncated or zero-sub-token words) are predicted 'O' and
    counted, so gold entities are never silently removed."""
    model.eval()
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)
    preds, missing, i = [], 0, 0
    for b in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=AMP_ON):
            logits = model(input_ids=b['input_ids'].to(device),
                           attention_mask=b['attention_mask'].to(device)).logits
        arg = logits.argmax(-1).cpu().numpy()
        for row in arg:
            n_words = len(ds.s[i]['tokens'])
            p = ['O'] * n_words; firsts = ds.first_positions(i)
            for j, w in firsts:
                p[w] = idx2tag[int(row[j])]
            missing += n_words - len(firsts)
            preds.append(p); i += 1
    return preds, missing


def sentence_counts(sents, preds, lexicon):
    """Per-sentence TP/FP/FN: columns = overall(3), seen(3), unseen(3).
    Gold entities are stratified by their own surface form, unmatched
    predictions by the surface form of the prediction."""
    C = np.zeros((len(sents), 9), dtype=np.int16)
    for i, (s, p) in enumerate(zip(sents, preds)):
        tok = s['tokens']
        G = set(iob_spans(s['tags'])); P = set(iob_spans(p))
        for sp in G:
            k = 3 if ' '.join(tok[sp[0]:sp[1]]).lower() in lexicon else 6
            hit = sp in P
            C[i, 0 if hit else 2] += 1; C[i, k if hit else k + 2] += 1
        for sp in P - G:
            k = 3 if ' '.join(tok[sp[0]:sp[1]]).lower() in lexicon else 6
            C[i, 1] += 1; C[i, k + 1] += 1
    return C


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def seqeval_check(gold, pred):
    try:
        from seqeval.metrics import f1_score
        return float(f1_score(gold, pred))
    except Exception:
        return None


def param_hash(model):
    h = hashlib.md5()
    for _, p in sorted(model.state_dict().items()):
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def determinism_check(train_all, tokenizer, tag2idx, collator, n_sent=64, steps=30):
    """Train the same short run twice and compare all weights bit for bit.
    Also reports every determinism warning raised by PyTorch."""
    sents = train_all[:n_sent]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hashes = []
        for _ in range(2):
            m, *_ = train(sents, 12345, steps, tokenizer, tag2idx, collator)
            hashes.append(param_hash(m)); del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    msgs = sorted({str(w.message).split('\n')[0][:160] for w in caught
                   if 'determin' in str(w.message).lower()})
    ok = hashes[0] == hashes[1]
    print(f'determinism check ({steps} steps, twice): {"PASS" if ok else "FAIL"}')
    for m in msgs:
        print('   torch warning:', m)
    return dict(passed=ok, steps=steps, warnings=msgs)


# =============================================================================
# RUN PLAN
# =============================================================================
def budget_steps(budget, s_b5, s_a5):
    if budget == 'match':
        return s_a5
    return int(budget[:-1]) * s_b5


def planned_runs(phase, label, seed):
    if phase == 1:
        return [(m, b, 0) for m, b in PRIMARY_RUNS]
    if phase == 2:
        prim = set(PRIMARY_RUNS)
        return [(m, b, 0) for m in SWEEP_METHODS for b in SWEEP_BUDGETS[label]
                if (m, b) not in prim]
    if phase == 3 and label in REPEAT_CELLS and seed in REPEAT_SEEDS:
        return [(m, b, 1) for m, b in REPEAT_RUNS]
    return []


FIELDS = ['corpus', 'few_shot', 'seed', 'method', 'budget', 'config', 'legacy',
          'repeat', 'phase',
          'f1', 'precision', 'recall',
          'f1_seen', 'p_seen', 'r_seen', 'n_seen',
          'f1_unseen', 'p_unseen', 'r_unseen', 'n_unseen',
          'dev_f1', 'dev_precision', 'dev_recall',
          'n_few', 'n_elig', 'n_dev', 'n_train', 'n_aug',
          'steps_per_epoch', 'n_steps', 'epochs_eff', 's_base5', 's_aug5', 'budget_mult',
          'lex_size', 'cov_test', 'subset_hash', 'train_hash',
          'amp', 'n_skipped', 'test_words_unlabelled', 'seqeval_f1',
          'runtime_s', 'session', 'gpu', 'code_version']


# =============================================================================
# OUTPUT MANAGEMENT (resume across Kaggle sessions)
# =============================================================================
def out_dir(slug):
    d = os.path.join(OUT_DIR, f'steps_{slug}'); os.makedirs(d, exist_ok=True)
    return d


def merge_npz(dst, src):
    data = dict(np.load(dst)) if os.path.exists(dst) else {}
    with np.load(src) as z:
        for k in z.files:
            data.setdefault(k, z[k])
    tmp = dst + '.tmp.npz'; np.savez_compressed(tmp, **data); os.replace(tmp, dst)


def import_previous(slug, od):
    """Merge outputs of earlier sessions found under INPUT_ROOT."""
    name = f'steps_{slug}'; found = 0
    if not os.path.isdir(INPUT_ROOT):
        return
    for root, _d, files in os.walk(INPUT_ROOT):
        if os.path.basename(root) != name or os.path.abspath(root) == os.path.abspath(od):
            continue
        for f in files:
            src = os.path.join(root, f); dst = os.path.join(od, f); found += 1
            if f.endswith('.npz'):
                merge_npz(dst, src)
            elif f in ('results.csv', 'aug_stats.csv'):
                frames = [pd.read_csv(src)] + ([pd.read_csv(dst)] if os.path.exists(dst) else [])
                pd.concat(frames).drop_duplicates().to_csv(dst, index=False)
            elif f == 'meta.json':
                prev = json.load(open(src))
                cur = json.load(open(dst)) if os.path.exists(dst) else {'sessions': []}
                ids = {s['session'] for s in cur.get('sessions', [])}
                cur['sessions'] = [s for s in prev.get('sessions', [])
                                   if s['session'] not in ids] + cur.get('sessions', [])
                json.dump(cur, open(dst, 'w'), indent=1)
            elif not os.path.exists(dst):
                shutil.copy(src, dst)
    if found:
        print(f'imported {found} files from earlier sessions')


def write_meta(od, reps, tokenizer, session, det=None):
    try:
        commit = getattr(AutoConfig.from_pretrained(MODEL_NAME, revision=MODEL_REVISION),
                         '_commit_hash', None)
    except Exception:
        commit = None
    try:
        import scipy, sklearn
        extra = {'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    except Exception:
        extra = {}
    sess = dict(
        session=session, started=time.strftime('%Y-%m-%d %H:%M:%S'),
        host=socket.gethostname(), python=platform.python_version(),
        platform=platform.platform(), torch=torch.__version__,
        transformers=transformers.__version__, numpy=np.__version__,
        pandas=pd.__version__, **extra,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        model=MODEL_NAME, model_commit=commit,
        tokenizer=dict(cls=type(tokenizer).__name__, is_fast=tokenizer.is_fast,
                       vocab_size=tokenizer.vocab_size,
                       do_lower_case=getattr(tokenizer, 'do_lower_case', None)),
        protocol=dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS,
                      eps=ADAM_EPS, grad_clip=GRAD_CLIP, batch=BATCH_SIZE,
                      max_seq_len=MAX_SEQ_LEN, schedule='linear decay to 0, no warm-up',
                      ref_epochs=REF_EPOCHS, K=AUG_K, rho=RHO, seeds=SEEDS,
                      fractions=FRACTIONS, primary=PRIMARY_RUNS,
                      sweep_methods=SWEEP_METHODS, sweep_budgets=SWEEP_BUDGETS,
                      amp=AMP_ON, deterministic=True, attn_implementation=ATTN_IMPL,
                      code_version=CODE_VERSION,
                      checkpoint_selection='none (fixed integer step budget)'),
        determinism_check=det,
        data=reps)
    p = os.path.join(od, 'meta.json')
    cur = json.load(open(p)) if os.path.exists(p) else {'sessions': []}
    cur['sessions'].append(sess)
    json.dump(cur, open(p, 'w'), indent=1, default=str)


def load_done(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(FIELDS)
        return set()
    df = pd.read_csv(path)
    if list(df.columns) != FIELDS:            # file from an older code version
        df = df.reindex(columns=FIELDS)
        df.to_csv(path, index=False)
    old = int((df.code_version.astype(str) != CODE_VERSION).sum())
    if old:
        print(f'{old} rows from older code versions are kept in the file but ignored')
    df = df[df.code_version.astype(str) == CODE_VERSION]
    return {(r.few_shot, r.config, int(r.seed), int(r.repeat)) for r in df.itertuples()}


def save_counts(path, key, C):
    data = dict(np.load(path)) if os.path.exists(path) else {}
    data[key] = C
    tmp = path + '.tmp.npz'; np.savez_compressed(tmp, **data); os.replace(tmp, path)


# =============================================================================
# CELL PREPARATION (subset, dev set, lexicon, augmented sets with cache)
# =============================================================================
class Cell:
    pass


def prepare_cell(data, frac, label, seed, od, tokenizer, get_encoder, slug, corpus):
    c = Cell()
    train_all, val_all = data['train'], data['validation']
    c.subset = [train_all[i] for i in few_shot_indices(len(train_all), frac, seed)]
    c.dev = [val_all[i] for i in dev_indices(len(val_all), len(c.subset), seed)]
    c.eligible = [s for s in c.subset if mentions_of(s)]
    c.lexicon = {' '.join(s['tokens'][a:b]).lower()
                 for s in c.subset for (a, b, _) in iob_spans(s['tags'])}
    c.s_b5 = math.ceil(len(c.subset) / BATCH_SIZE) * REF_EPOCHS
    c.s_a5 = math.ceil((len(c.subset) + len(c.eligible) * AUG_K) / BATCH_SIZE) * REF_EPOCHS
    c.subset_hash = md5_obj([s['id'] for s in c.subset])
    c.aug = None
    path = os.path.join(od, f'aug_{cell_tag(label)}_s{seed}.json.gz')
    c.aug_path = path
    if os.path.exists(path):
        with gzip.open(path, 'rt') as f:
            cache = json.load(f)
        assert cache['subset_ids'] == [s['id'] for s in c.subset], \
            f'cached subset differs from recomputed subset: {path}'
        if cache.get('code_version') == CODE_VERSION:
            c.aug = cache['aug']
    return c


def ensure_aug(c, label, seed, od, tokenizer, get_encoder, slug, corpus):
    """MR and COSINER sets are built once per (cell, seed), written to disk and
    reused for every budget, so all budgets of a method see the same data."""
    if c.aug is not None:
        return
    rank = build_rankings(c.subset, tokenizer, get_encoder())
    cand = [len(v) for v in rank.values()] or [0]
    sets, rows = {}, []
    for method, rk in [('COSINER', rank), ('MR', shuffled_rankings(rank, seed))]:
        aug, st = augment(c.eligible, rk, method.lower())
        sets[method] = aug
        rows.append(dict(corpus=corpus, few_shot=label, seed=seed, method=method,
                         n_few=len(c.subset), n_elig=len(c.eligible),
                         lex_size_cased=len(rank), lex_size=len(c.lexicon),
                         cand_mean=float(np.mean(cand)), cand_min=int(np.min(cand)),
                         n_aug=len(aug), **st))
    rows.append(dict(corpus=corpus, few_shot=label, seed=seed, method='Dup',
                     n_few=len(c.subset), n_elig=len(c.eligible),
                     lex_size_cased=len(rank), lex_size=len(c.lexicon),
                     cand_mean=np.nan, cand_min=np.nan, n_aug=len(c.eligible) * AUG_K,
                     copies_target=len(c.eligible) * AUG_K,
                     copies_made=len(c.eligible) * AUG_K, attempts=0, failed_attempts=0,
                     sent_short=0, rep_per_copy=0.0, frac_mentions_replaced=0.0,
                     mean_abs_dlen=0.0))
    sp = os.path.join(od, 'aug_stats.csv')
    df = pd.DataFrame(rows)
    df.to_csv(sp, mode='a', header=not os.path.exists(sp), index=False)
    c.aug = sets
    if EXPORT_AUG:
        with gzip.open(c.aug_path, 'wt') as f:
            json.dump({'code_version': CODE_VERSION,
                       'subset_ids': [s['id'] for s in c.subset],
                       'dev_ids': [s['id'] for s in c.dev],
                       'aug': sets}, f)


def training_set(c, method):
    if method == 'Base':
        return list(c.subset), 0
    if method == 'Dup':
        aug = dup_set(c.eligible)
    else:
        aug = c.aug[method]
    return list(c.subset) + aug, len(aug)


# =============================================================================
# DRY RUN
# =============================================================================
def dry_run(data, corpus):
    print(f'\nRUN PLAN for {corpus} (no training)')
    tot_runs, tot_steps = Counter(), Counter()
    rows = []
    for frac, label in active_fractions():
        for seed in active_seeds():
            n = len(few_shot_indices(len(data['train']), frac, seed))
            sub = [data['train'][i] for i in few_shot_indices(len(data['train']), frac, seed)]
            ne = sum(1 for s in sub if mentions_of(s))
            s_b5 = math.ceil(n / BATCH_SIZE) * REF_EPOCHS
            s_a5 = math.ceil((n + ne * AUG_K) / BATCH_SIZE) * REF_EPOCHS
            rows.append(dict(cell=label, seed=seed, n=n, n_elig=ne, s_b5=s_b5,
                             s_a5=s_a5, ratio=s_a5 / s_b5))
            for ph in PHASES:
                for m, b, r in planned_runs(ph, label, seed):
                    tot_runs[ph] += 1; tot_steps[ph] += budget_steps(b, s_b5, s_a5)
    df = pd.DataFrame(rows)
    print(df.groupby('cell').agg(n=('n', 'mean'), n_elig=('n_elig', 'mean'),
                                 s_b5=('s_b5', 'mean'), s_a5=('s_a5', 'mean'),
                                 ratio_min=('ratio', 'min'),
                                 ratio_max=('ratio', 'max')).round(2).to_string())
    for ph in PHASES:
        print(f'  phase {ph}: {tot_runs[ph]:5d} runs, {tot_steps[ph]:9d} optimizer steps')
    print(f'  total  : {sum(tot_runs.values()):5d} runs, {sum(tot_steps.values()):9d} '
          f'optimizer steps (+ one test and one dev evaluation per run)')


def active_fractions():
    return [(f, l) for f, l in FRACTIONS if l in CELLS_TO_RUN]


def active_seeds():
    return [s for s in SEEDS if SEED_SHARD is None or s in SEED_SHARD]


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    spec = CORPORA[CORPUS]; slug = spec['slug']; ent = spec['entity']
    print(f'corpus={CORPUS} device={device} amp={AMP_ON} seeds={active_seeds()} '
          f'phases={PHASES} cells={CELLS_TO_RUN}')
    data, reps = load_corpus(CORPUS)
    print(f"=== {CORPUS}: train={len(data['train'])} val={len(data['validation'])} "
          f"test={len(data['test'])}")
    if DRY_RUN:
        dry_run(data, CORPUS); return

    od = out_dir(slug)
    import_previous(slug, od)
    session = time.strftime('%Y%m%d-%H%M%S') + '-' + hashlib.md5(
        str(random.random()).encode()).hexdigest()[:6]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    ent_tags = {'O': 0, 'B-' + ent: 1, 'I-' + ent: 2}
    det = determinism_check(data['train'], tokenizer, ent_tags,
                            DataCollatorForTokenClassification(tokenizer,
                                                               label_pad_token_id=-100))
    write_meta(od, reps, tokenizer, session, det)
    if STRICT_DETERMINISM and not det['passed']:
        raise RuntimeError('Determinism check failed; results would not be reproducible. '
                           'Send the printed warnings for diagnosis.')
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'

    tag2idx = {'O': 0, 'B-' + ent: 1, 'I-' + ent: 2}
    idx2tag = {v: k for k, v in tag2idx.items()}
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    test_s = data['test']
    test_ds = NERDataset(test_s, tokenizer, tag2idx, eval_mode=True)
    test_surf = [' '.join(s['tokens'][a:b]).lower()
                 for s in test_s for (a, b, _) in iob_spans(s['tags'])]

    res_path = os.path.join(od, 'results.csv')
    done = load_done(res_path)
    print(f'finished runs found: {len(done)}')

    enc_holder = {}
    def get_encoder():
        if 'enc' not in enc_holder:
            enc_holder['enc'] = AutoModel.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION,
                attn_implementation=ATTN_IMPL).to(device).eval()
        return enc_holder['enc']

    for phase in PHASES:
        for frac, label in active_fractions():
            for seed in active_seeds():
                todo = [r for r in planned_runs(phase, label, seed)
                        if (label, f'{r[0]}@{r[1]}', seed, r[2]) not in done]
                if not todo:
                    continue
                c = prepare_cell(data, frac, label, seed, od, tokenizer, get_encoder,
                                 slug, CORPUS)
                if any(m in ('MR', 'COSINER') for m, _, _ in todo):
                    ensure_aug(c, label, seed, od, tokenizer, get_encoder, slug, CORPUS)
                cov = float(np.mean([m in c.lexicon for m in test_surf])) if test_surf else 0.0
                dev_ds = NERDataset(c.dev, tokenizer, tag2idx, eval_mode=True)
                cpath = os.path.join(od, f'counts_{cell_tag(label)}_s{seed}.npz')
                print(f'\n[phase {phase}] {label} seed={seed}: n={len(c.subset)} '
                      f'eligible={len(c.eligible)} |L|={len(c.lexicon)} cov={cov:.3f} '
                      f'S_b5={c.s_b5} S_a5={c.s_a5} (ratio {c.s_a5 / c.s_b5:.2f})')

                for method, budget, rep in todo:
                    if (time.time() - t0) / 3600 > TIME_BUDGET_H:
                        print('\ntime budget reached; stopping cleanly. Re-run with this '
                              'output attached as Input to continue.')
                        return
                    t1 = time.time()
                    cfg = f'{method}@{budget}'
                    steps = budget_steps(budget, c.s_b5, c.s_a5)
                    tr, n_aug = training_set(c, method)
                    model, done_steps, skipped, spe = train(tr, seed, steps, tokenizer,
                                                            tag2idx, collator)
                    assert done_steps == steps, (cfg, done_steps, steps)
                    preds, missing = predict(model, test_ds, collator, idx2tag)
                    C = sentence_counts(test_s, preds, c.lexicon)
                    tot = C.sum(0).astype(int)
                    p, r, f1 = prf(*tot[0:3]); ps, rs, fs = prf(*tot[3:6])
                    pu, ru, fu = prf(*tot[6:9])
                    sq = seqeval_check([s['tags'] for s in test_s], preds)
                    if sq is not None and abs(sq - f1) > 1e-9:
                        print(f'  WARNING: seqeval F1 {sq:.6f} != internal F1 {f1:.6f}')
                    dpred, _ = predict(model, dev_ds, collator, idx2tag)
                    Cd = sentence_counts(c.dev, dpred, c.lexicon).sum(0).astype(int)
                    dp, dr, df1 = prf(*Cd[0:3])
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_counts(cpath, f'{cfg}__r{rep}', C)
                    row = dict(corpus=CORPUS, few_shot=label, seed=seed, method=method,
                               budget=budget, config=cfg, legacy=LEGACY.get(cfg, ''),
                               repeat=rep, phase=phase,
                               f1=f1, precision=p, recall=r,
                               f1_seen=fs, p_seen=ps, r_seen=rs, n_seen=int(tot[3] + tot[5]),
                               f1_unseen=fu, p_unseen=pu, r_unseen=ru,
                               n_unseen=int(tot[6] + tot[8]),
                               dev_f1=df1, dev_precision=dp, dev_recall=dr,
                               n_few=len(c.subset), n_elig=len(c.eligible), n_dev=len(c.dev),
                               n_train=len(tr), n_aug=n_aug, steps_per_epoch=spe,
                               n_steps=done_steps, epochs_eff=done_steps / spe,
                               s_base5=c.s_b5, s_aug5=c.s_a5,
                               budget_mult=done_steps / c.s_b5,
                               lex_size=len(c.lexicon), cov_test=cov,
                               subset_hash=c.subset_hash,
                               train_hash=md5_obj([[s['id'], s['tokens'], s['tags']] for s in tr]),
                               amp=AMP_ON, n_skipped=skipped, test_words_unlabelled=missing,
                               seqeval_f1=sq if sq is not None else '',
                               runtime_s=round(time.time() - t1, 1), session=session, gpu=gpu,
                               code_version=CODE_VERSION)
                    with open(res_path, 'a', newline='') as f:
                        csv.writer(f).writerow([row[k] for k in FIELDS])
                    done.add((label, cfg, seed, rep))
                    print(f'  {cfg:15s} r{rep} n_train={len(tr):5d} steps={done_steps:5d} '
                          f'ep={done_steps / spe:6.2f} F1={f1 * 100:5.2f} '
                          f'dev={df1 * 100:5.2f} seen={fs * 100:5.2f} unseen={fu * 100:5.2f} '
                          f'({time.time() - t1:.0f}s)')
    print(f'\nall planned runs finished. Outputs: {od}')


if __name__ == '__main__':
    main()
