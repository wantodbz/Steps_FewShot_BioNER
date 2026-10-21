# =============================================================================
#  check_truncation.py — audit of the 512-position input limit (MAX_SEQ_LEN)
#
#  The experiment scripts encode every sample with
#      tokenizer(words, is_split_into_words=True, max_length=512, truncation=True)
#  so a sample longer than 512 positions ([CLS] + WordPiece sub-tokens + [SEP])
#  would lose its trailing sub-tokens. This script measures whether that ever
#  happened. For every corpus, few-shot cell and seed it rebuilds
#    * the training sets of all configurations, exactly as steps_<corpus>.py
#      builds them: the few-shot subset (Base-E5, Base-match and the Base
#      sweep), subset + K exact copies (Duplicate), and subset + the stored
#      MR and COSINER sets (aug_<cell>_s<seed>.json.gz),
#    * the few-shot development set, and
#    * the full test set,
#  encodes every sample WITHOUT truncation and counts the samples that exceed
#  512 positions, the words and entity mentions that truncation would affect,
#  and the longest sample. The rebuilt training sets are checked against the
#  subset and training-set hashes recorded in results.csv, so the audit refers
#  to exactly the data that were used in training.
#
#  Usage (CPU, about one minute), after the preparation step of the README,
#  which copies Dataset/ and result_<corpus>_<cell>/ to data/ and outputs/:
#    python check_truncation.py --data ./data --outputs ./outputs --out ./analysis
#  Below outputs/steps_<corpus>/, results.csv and aug_*.json.gz files are found
#  in any sub-folder (one per few-shot cell after the preparation step).
#  Tokenizer: by default the Hugging Face tokenizer dmis-lab/biobert-v1.1 at
#  the revision used in the experiments (needs `transformers`, not PyTorch);
#  --tokenizer accepts another name or a local folder. Offline:
#    python check_truncation.py --vocab /path/to/biobert_v1.1/vocab.txt
#  uses the BioBERT v1.1 vocabulary with the same cased WordPiece settings
#  (needs only `tokenizers`); both routes give identical encodings.
#  Writes table_truncation.csv (per corpus, cell and set) and
#  table_truncation_summary.csv (per corpus and set).
# =============================================================================
import argparse, glob, gzip, hashlib, json, os, random, re
from collections import Counter

import pandas as pd

MAX_SEQ_LEN = 512
AUG_K = 5
SEEDS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
FRACTIONS = [(0.02, '2%'), (0.05, '5%'), (0.10, '10%')]
MODEL_NAME = 'dmis-lab/biobert-v1.1'
MODEL_REVISION = '551ca18efd7f052c8dfa0b01c94c2a8e68bc5488'
CORPORA = {
    'NCBI-Disease': dict(kind='generic', entity='Disease', slug='ncbi', dir='ncbi-disease'),
    'BC5CDR-Chem':  dict(kind='bc5cdr',  entity='Chemical', slug='bc5cdr', dir='bc5cdr'),
    'BC2GM':        dict(kind='generic', entity='Gene', slug='bc2gm', dir='bc2gm-corpus'),
}
SETS = ['train: original', 'train: Duplicate', 'train: MR', 'train: COSINER', 'dev', 'test']
METHOD_OF_SET = {'train: original': 'Base', 'train: Duplicate': 'Dup',
                 'train: MR': 'MR', 'train: COSINER': 'COSINER'}


# ---- identical to steps_<corpus>.py (code version 2.1) ----------------------
def parse_token_list(cell):
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
    if kind == 'bc5cdr':
        tag_map = {0: 'O', 1: 'B-' + entity, 2: 'O', 3: 'O', 4: 'I-' + entity}
    else:
        tag_map = {0: 'O', 1: 'B-' + entity, 2: 'I-' + entity}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    tok_col = 'tokens' if 'tokens' in df.columns else df.columns[0]
    tag_col = 'ner_tags' if 'ner_tags' in df.columns else df.columns[1]
    ids = df['id'].tolist() if 'id' in df.columns else [''] * len(df)
    out = []
    for ri, (tcell, gcell, rid) in enumerate(zip(df[tok_col].tolist(),
                                                 df[tag_col].tolist(), ids)):
        tokens = parse_token_list(tcell)
        tags = [tag_map.get(t, 'O') for t in parse_tag_list(gcell)]
        rid = rid if str(rid) != '' else ri
        if tokens and len(tokens) == len(tags):
            out.append({'id': f'{split}:{rid}', 'tokens': tokens, 'tags': tags})
    return out


def iob_spans(tags):
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


def shuffled_prefix(n_total, k, seed):
    rng = random.Random(seed)
    idx = list(range(n_total)); rng.shuffle(idx)
    return idx[:k]


def few_shot_indices(n_total, frac, seed):
    return shuffled_prefix(n_total, max(1, int(n_total * frac)), seed)


def dev_indices(n_val, n_dev, seed):
    return shuffled_prefix(n_val, min(n_dev, n_val), 1_000_003 + seed)


def dup_set(eligible):
    return [{'id': f"{s['id']}#dup{k}", 'tokens': list(s['tokens']), 'tags': list(s['tags'])}
            for s in eligible for k in range(AUG_K)]


def md5_obj(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=False).encode()).hexdigest()


def cell_tag(label):
    return {'2%': '02pct', '5%': '05pct', '10%': '10pct'}[label]


# ---- tokenizer ---------------------------------------------------------------
class Encoder:
    """Returns, for a list of words, the number of input positions including
    [CLS] and [SEP] and the word index of every position (None for specials),
    computed WITHOUT truncation."""

    def __init__(self, vocab=None, tokenizer=MODEL_NAME):
        if vocab:
            from tokenizers import BertWordPieceTokenizer
            self.tok = BertWordPieceTokenizer(vocab, clean_text=True, handle_chinese_chars=True,
                                              strip_accents=None, lowercase=False)
            self.kind = f'tokenizers.BertWordPieceTokenizer({os.path.abspath(vocab)}, cased)'
        else:
            from transformers import AutoTokenizer
            rev = MODEL_REVISION if tokenizer == MODEL_NAME else None
            self.tok = AutoTokenizer.from_pretrained(tokenizer, revision=rev)
            self.kind = (f'transformers {type(self.tok).__name__} {tokenizer}'
                         + (f'@{rev[:7]}' if rev else '')
                         + f' (vocab {self.tok.vocab_size}, lowercase '
                         + f"{getattr(self.tok, 'do_lower_case', None)})")
        self.vocab = vocab
        self.cache = {}

    def encode(self, words):
        key = tuple(words)
        if key not in self.cache:
            if self.vocab:
                e = self.tok.encode(list(words), is_pretokenized=True, add_special_tokens=True)
                wid = list(e.word_ids)
            else:
                e = self.tok(list(words), is_split_into_words=True, truncation=False)
                wid = e.word_ids()
            self.cache[key] = wid
        return self.cache[key]


def audit(sample, enc):
    """Positions, truncation flag, words without any kept sub-token, and
    entity mentions that would lose at least one sub-token."""
    wid = enc.encode(sample['tokens'])
    n_pos = len(wid)
    if n_pos <= MAX_SEQ_LEN:
        return n_pos, False, 0, 0
    kept = wid[:MAX_SEQ_LEN - 1]                   # the last kept position is [SEP]
    kept_words = {w for w in kept if w is not None}
    last_full = max(kept_words) if kept_words else -1
    lost_words = len(sample['tokens']) - len(kept_words)
    cut = [w for w in wid[MAX_SEQ_LEN - 1:] if w is not None]
    first_cut_word = min(cut) if cut else len(sample['tokens'])
    hit = sum(1 for (st, en, _) in iob_spans(sample['tags']) if en - 1 >= first_cut_word)
    return n_pos, True, lost_words, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder with <corpus>/train.csv, validation.csv, test.csv')
    ap.add_argument('--outputs', default='outputs', help='folder with steps_<corpus>/ experiment outputs')
    ap.add_argument('--out', default='analysis', help='folder for the output tables')
    ap.add_argument('--tokenizer', default=MODEL_NAME, help='Hugging Face name or local folder of the tokenizer')
    ap.add_argument('--vocab', default=None, help='offline: BioBERT v1.1 vocab.txt instead of the HF tokenizer')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = Encoder(a.vocab, a.tokenizer)
    print(f'tokenizer: {enc.kind}; limit {MAX_SEQ_LEN} positions incl. [CLS] and [SEP]')

    rows, checks = [], Counter()
    for corpus, spec in CORPORA.items():
        d = os.path.join(a.data, spec['dir'])
        data = {sp: parse_file(os.path.join(d, f'{sp}.csv'), spec['kind'], spec['entity'], sp)
                for sp in ('train', 'validation', 'test')}
        by_id = {sp: {s['id']: s for s in v} for sp, v in data.items()}
        od = os.path.join(a.outputs, f"steps_{spec['slug']}")
        rfiles = sorted(glob.glob(os.path.join(od, '**', 'results.csv'), recursive=True))
        if not rfiles:
            raise FileNotFoundError(f'no results.csv below {od}; run the preparation step of the README first')
        R = pd.concat([pd.read_csv(f) for f in rfiles], ignore_index=True)
        R = R[R.code_version.astype(str) == '2.1']

        # test set (shared by all cells and runs)
        stat = [audit(s, enc) for s in data['test']]
        rows.append(dict(corpus=corpus, few_shot='all', set='test', n_sets=1,
                         n_samples=len(stat), n_truncated=sum(t for _, t, _, _ in stat),
                         words_lost=sum(w for *_, w, _ in stat), mentions_hit=sum(h for *_, h in stat),
                         max_positions=max(p for p, *_ in stat)))
        for frac, label in FRACTIONS:
            acc = {k: Counter() for k in SETS[:-1]}
            for seed in SEEDS:
                hits = glob.glob(os.path.join(od, '**', f'aug_{cell_tag(label)}_s{seed}.json.gz'),
                                 recursive=True)
                if len(hits) != 1:
                    raise FileNotFoundError(f'aug_{cell_tag(label)}_s{seed}.json.gz below {od}: {hits}')
                with gzip.open(hits[0], 'rt') as f:
                    cache = json.load(f)
                subset = [by_id['train'][i] for i in cache['subset_ids']]
                dev = [by_id['validation'][i] for i in cache['dev_ids']]
                # the stored IDs must equal the IDs that the sampling code draws
                checks['subset_ids'] += [data['train'][i]['id'] for i in
                                         few_shot_indices(len(data['train']), frac, seed)] == cache['subset_ids']
                checks['dev_ids'] += [data['validation'][i]['id'] for i in
                                      dev_indices(len(data['validation']), len(subset), seed)] == cache['dev_ids']
                checks['cells'] += 1
                eligible = [s for s in subset if iob_spans(s['tags'])]
                train_sets = {'train: original': list(subset),
                              'train: Duplicate': list(subset) + dup_set(eligible),
                              'train: MR': list(subset) + cache['aug']['MR'],
                              'train: COSINER': list(subset) + cache['aug']['COSINER']}
                rr = R[(R.few_shot == label) & (R.seed == seed)]
                sub_hash = md5_obj([s['id'] for s in subset])
                checks['subset_hash_ok'] += int((rr.subset_hash == sub_hash).all())
                for name, tr in train_sets.items():
                    runs = rr[rr.method == METHOD_OF_SET[name]]
                    h = md5_obj([[s['id'], s['tokens'], s['tags']] for s in tr])
                    checks['runs'] += len(runs)
                    checks['train_hash_ok'] += int((runs.train_hash == h).sum())
                    checks['n_train_ok'] += int((runs.n_train == len(tr)).sum())
                for name, ss in list(train_sets.items()) + [('dev', dev)]:
                    st = [audit(s, enc) for s in ss]
                    c = acc[name]; c['n_sets'] += 1; c['n_samples'] += len(st)
                    c['n_truncated'] += sum(t for _, t, _, _ in st)
                    c['words_lost'] += sum(w for *_, w, _ in st)
                    c['mentions_hit'] += sum(h for *_, h in st)
                    c['max_positions'] = max(c['max_positions'], max(p for p, *_ in st))
            for name in SETS[:-1]:
                rows.append(dict(corpus=corpus, few_shot=label, set=name, **acc[name]))

    T = pd.DataFrame(rows)[['corpus', 'few_shot', 'set', 'n_sets', 'n_samples', 'n_truncated',
                             'words_lost', 'mentions_hit', 'max_positions']]
    T['pct_truncated'] = 100 * T.n_truncated / T.n_samples
    T.to_csv(os.path.join(a.out, 'table_truncation.csv'), index=False)
    S = (T.groupby(['corpus', 'set'], sort=False)
          .agg(n_samples=('n_samples', 'sum'), n_truncated=('n_truncated', 'sum'),
               words_lost=('words_lost', 'sum'), mentions_hit=('mentions_hit', 'sum'),
               max_positions=('max_positions', 'max')).reset_index())
    S['pct_truncated'] = 100 * S.n_truncated / S.n_samples
    S.to_csv(os.path.join(a.out, 'table_truncation_summary.csv'), index=False)

    print(f"\nintegrity: {checks['subset_ids']}/{checks['cells']} subsets and "
          f"{checks['dev_ids']}/{checks['cells']} dev sets match the sampling code; "
          f"subset hash ok in {checks['subset_hash_ok']}/{checks['cells']} (cell, seed) pairs; "
          f"training-set hash ok in {checks['train_hash_ok']}/{checks['runs']} runs; "
          f"n_train ok in {checks['n_train_ok']}/{checks['runs']} runs")
    print('\nSamples longer than 512 positions (summed over cells and seeds; '
          'training sets are counted once per cell and seed, not once per budget):')
    print(S.to_string(index=False))
    print(f"\nlongest sample overall: {int(T.max_positions.max())} positions "
          f"(limit {MAX_SEQ_LEN}); truncated samples overall: {int(T.n_truncated.sum())}")


if __name__ == '__main__':
    main()
