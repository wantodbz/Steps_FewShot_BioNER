# =============================================================================
#  analysis_steps_v2.py  —  statistical analysis for the revised experiments
#  CPU only, no training. Reads the outputs of steps_experiment.py.
#
#  Kaggle: new notebook, Accelerator None, add the output notebook(s) of
#  steps_experiment.py as Input, paste this file into one cell, run.
#
#  What changed relative to the first submission (reviewer/editor requests):
#   * Per-cell tests: Wilcoxon signed-rank for the difference AND a Wilcoxon-
#     based TOST for equivalence (one framework); paired t versions are given
#     as sensitivity. Holm adjustment is applied to BOTH within each family,
#     and the verdict is derived from the adjusted p-values only.
#   * The nine cells are NOT treated as independent. Within a corpus the
#     few-shot subsets are nested (same seed -> 2% c 5% c 10%) and all cells
#     share one test set. Conclusions are therefore drawn PER CORPUS with a
#     two-way cluster bootstrap that resamples seeds (each seed carries its
#     three nested cells) and test sentences (shared by all cells/configs).
#     A linear mixed model with a random seed intercept is a sensitivity
#     analysis. The nine-cell summary is reported as descriptive only.
#   * Equivalence: TOST at the pre-specified margin of 1.0 F1 point, plus a
#     sensitivity analysis over margins 0.5 / 1.0 / 1.5.
#   * Training-budget sweep (R1.1): curves for Base, MR and COSINER at
#     identical integer step budgets; per-budget paired differences;
#     best-of-curve comparison with the budget selected on a few-shot dev set
#     (practical) and with the test-set peak (oracle, exploratory only).
#   * Integrity checks: exact step parity, identical training sets across
#     budgets, run-to-run repeatability, metric cross-checks.
# =============================================================================

import os, glob, json, re, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOTS = os.environ.get('STEPS_ANALYSIS_ROOTS', '/kaggle/input:/kaggle/working:.').split(':')
OUT = os.environ.get('STEPS_ANALYSIS_OUT',
                     '/kaggle/working/analysis' if os.path.isdir('/kaggle/working')
                     else './analysis')

EQUIV_MARGIN = 1.0              # pre-specified before the first experiments
SENS_MARGINS = [0.5, 1.0, 1.5]  # sensitivity only
ALPHA = 0.05
N_BOOT = int(os.environ.get('STEPS_N_BOOT', 2000))
CODE_VERSION = os.environ.get('STEPS_CODE_VERSION', '2.1')  # results of other versions are ignored
BOOT_SEED = 20261001

PRIMARY = [('MR@match', 'Base@1x'), ('MR@match', 'Dup@match'),
           ('MR@match', 'Base@match'), ('Dup@match', 'Base@match')]
SECONDARY = [('COSINER@match', 'Base@1x'), ('COSINER@match', 'Dup@match'),
             ('COSINER@match', 'Base@match')]
DECOMP = [('optimisation', 'Base@match', 'Base@1x'),
          ('content (MR)', 'MR@match', 'Base@match'),
          ('content (COSINER)', 'COSINER@match', 'Base@match'),
          ('total (MR)', 'MR@match', 'Base@1x')]
SWEEP_METHODS = ['Base', 'MR', 'COSINER']
CORPUS_ORDER = ['NCBI-Disease', 'BC5CDR-Chem', 'BC2GM']
CELL_ORDER = ['2%', '5%', '10%']
SLUG2CORPUS = {'ncbi': 'NCBI-Disease', 'bc5cdr': 'BC5CDR-Chem', 'bc2gm': 'BC2GM'}
TAG2CELL = {'02pct': '2%', '05pct': '5%', '10pct': '10%'}
METRIC_OFF = {'all': 0, 'seen': 3, 'unseen': 6}

os.makedirs(OUT, exist_ok=True)
_LOG = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); _LOG.append(s)


def H(title):
    P('\n' + '=' * 96); P(title); P('=' * 96)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)


# =============================================================================
# LOADING
# =============================================================================
SLUG_RE = re.compile(r'steps_(ncbi|bc5cdr|bc2gm)')


def _glob(fname):
    """All files with this name whose path contains a steps_<corpus> folder,
    whatever the folder or Kaggle dataset/notebook is called."""
    files = []
    for r in ROOTS:
        if os.path.isdir(r):
            files += glob.glob(os.path.join(r, '**', fname), recursive=True)
    return sorted({os.path.realpath(f) for f in files if SLUG_RE.search(f)})


def _corpus_of(path):
    return SLUG2CORPUS[SLUG_RE.findall(path)[-1]]


def load():
    rfiles = _glob('results.csv')
    if not rfiles:
        raise FileNotFoundError('No results.csv inside a steps_<corpus> folder found. '
                                'Attach the outputs of the experiment notebooks as Input.')
    R = pd.concat([pd.read_csv(f) for f in rfiles], ignore_index=True)
    ver = R['code_version'].astype(str) if 'code_version' in R else pd.Series('2.0', index=R.index)
    ver = ver.where(ver != 'nan', '2.0')
    n_old = int((ver != CODE_VERSION).sum())
    R = R[ver == CODE_VERSION]
    if n_old:
        print(f'ignored {n_old} rows from code versions other than {CODE_VERSION}')
    R = R.drop_duplicates(subset=['corpus', 'few_shot', 'config', 'seed', 'repeat'],
                          keep='last')

    # per-sentence counts: several copies of the same key can exist (sessions that
    # imported earlier outputs, older code versions). Keep the copy that
    # reproduces the F1 recorded in results.csv for the current code version.
    f1_of = {(r.corpus, r.few_shot, int(r.seed), f'{r.config}__r{int(r.repeat)}'): r.f1
             for r in R.itertuples()}
    cand = {}
    nfiles = _glob_counts()
    for f in nfiles:
        m = re.match(r'counts_(\w+)_s(\d+)\.npz', os.path.basename(f))
        corpus = _corpus_of(f); fs = TAG2CELL[m.group(1)]; seed = int(m.group(2))
        with np.load(f) as z:
            for k in z.files:
                cand.setdefault((corpus, fs, seed, k), []).append(z[k].astype(np.float32))
    counts, unmatched = {}, 0
    for (corpus, fs, seed, k), arrs in cand.items():
        want = f1_of.get((corpus, fs, seed, k))
        if want is None:
            continue                              # run not in the current version
        pick = None
        for a in arrs:
            t = a.sum(0)
            if abs(_metric(t[0], t[1], t[2], 'f1') - want) < 1e-9:
                pick = a; break
        if pick is None:
            unmatched += 1; continue
        counts.setdefault((corpus, fs, seed), {})[k] = pick
    if unmatched:
        print(f'WARNING: {unmatched} runs have no count array matching their F1')
    afiles = _glob('aug_stats.csv')
    A = (pd.concat([pd.read_csv(f) for f in afiles], ignore_index=True)
         .drop_duplicates(subset=['corpus', 'few_shot', 'seed', 'method'], keep='last')
         if afiles else pd.DataFrame())
    P(f'read {len(rfiles)} results files, {len(R)} runs, '
      f'{sum(len(v) for v in counts.values())} per-sentence count arrays '
      f'from {len(nfiles)} npz files')
    return R, counts, A


def _glob_counts():
    files = []
    for r in ROOTS:
        if os.path.isdir(r):
            files += glob.glob(os.path.join(r, '**', 'counts_*.npz'), recursive=True)
    return sorted({os.path.realpath(f) for f in files if SLUG_RE.search(f)})


def cells(R):
    out = []
    for c in CORPUS_ORDER:
        for fs in CELL_ORDER:
            if ((R.corpus == c) & (R.few_shot == fs)).any():
                out.append((c, fs))
    return out


def piv(R, c, fs, metric='f1'):
    sub = R[(R.corpus == c) & (R.few_shot == fs) & (R.repeat == 0)]
    return sub.pivot_table(index='seed', columns='config', values=metric)


# =============================================================================
# PER-CELL TESTS
# =============================================================================
def holm_adjust(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def _wil(x, alternative='two-sided'):
    x = np.asarray(x, float)
    if np.all(x == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative=alternative).pvalue)
    except ValueError:
        return 1.0


def paired_tests(d, delta=EQUIV_MARGIN):
    """d: per-seed differences in F1 points."""
    d = np.asarray(d, float); n = len(d); m = d.mean(); s = d.std(ddof=1)
    se = max(s / np.sqrt(n), 1e-12)
    tq95, tq90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(
        n=n, d=m, sd=s,
        ci95_lo=m - tq95 * se, ci95_hi=m + tq95 * se,
        ci90_lo=m - tq90 * se, ci90_hi=m + tq90 * se,
        p_diff=_wil(d),
        p_tost=max(_wil(d + delta, 'greater'), _wil(d - delta, 'less')),
        p_diff_t=float(2 * stats.t.sf(abs(m / se), n - 1)),
        p_tost_t=float(max(stats.t.sf((m + delta) / se, n - 1),
                           stats.t.cdf((m - delta) / se, n - 1))),
        pos=int((d > 0).sum()))


def verdict(p_diff, p_tost):
    a, b = p_diff < ALPHA, p_tost < ALPHA
    if a and b: return 'trivial'
    if a: return 'different'
    if b: return 'equivalent'
    return 'inconclusive'


def per_cell_family(R, family, fam_name, metric='f1'):
    rows = []
    for c, fs in cells(R):
        pv = piv(R, c, fs, metric)
        recs = []
        for a, b in family:
            if a not in pv or b not in pv:
                continue
            x = pv[[a, b]].dropna()
            if len(x) < 6:
                continue
            t = paired_tests((x[a] - x[b]).values * 100)
            recs.append(dict(corpus=c, few_shot=fs, family=fam_name,
                             comparison=f'{a} vs {b}', **t))
        if not recs:
            continue
        df = pd.DataFrame(recs)
        df['p_diff_holm'] = holm_adjust(df.p_diff)
        df['p_tost_holm'] = holm_adjust(df.p_tost)
        df['verdict'] = [verdict(x, y) for x, y in zip(df.p_diff_holm, df.p_tost_holm)]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# TWO-WAY CLUSTER BOOTSTRAP (seeds x test sentences), per corpus
# =============================================================================
class Boot:
    def __init__(self, n_test, n_seed, two_way=True, B=N_BOOT, seed=BOOT_SEED):
        rng = np.random.default_rng(seed)
        self.B, self.n_seed = B, n_seed
        self.S = rng.integers(0, n_seed, size=(B, n_seed))
        self.W = (rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=B)
                  .astype(np.float32) if two_way else None)


def _metric(tp, fp, fn, metric):
    with np.errstate(invalid='ignore', divide='ignore'):
        if metric == 'f1':
            v = 2 * tp / (2 * tp + fp + fn)
        elif metric == 'precision':
            v = tp / (tp + fp)
        else:
            v = tp / (tp + fn)
    return np.nan_to_num(v)


def _summed(T, boot):
    """T: (seed, sentence, 3) -> resampled sums (B, seed, 3)."""
    if boot is None:
        return T.sum(1)[None]
    S = (np.broadcast_to(T.sum(1)[None], (boot.B,) + T.sum(1).shape)
         if boot.W is None else np.tensordot(boot.W, T, axes=([1], [1])))
    return S[np.arange(boot.B)[:, None], boot.S]


def corpus_stat(TA, TB, boot, metric='f1', stratum='all'):
    """TA, TB: dict cell -> (seed, sentence, 9) arrays with aligned seeds.
    Returns the corpus-level mean difference (F1 points) over cells and seeds,
    for the original data (boot=None) or for every bootstrap draw."""
    o = METRIC_OFF[stratum]; per_cell = []
    for fs in TA:
        a = _summed(TA[fs][..., o:o + 3], boot); b = _summed(TB[fs][..., o:o + 3], boot)
        da = _metric(a[..., 0], a[..., 1], a[..., 2], metric)
        db = _metric(b[..., 0], b[..., 1], b[..., 2], metric)
        per_cell.append((da - db).mean(1))
    return np.mean(per_cell, axis=0) * 100


def tensors(counts, c, cell_list, seeds, chooser):
    """chooser(fs, seed) -> counts key. Returns dict cell -> (seed, sent, 9)."""
    out = {}
    for fs in cell_list:
        out[fs] = np.stack([counts[(c, fs, s)][chooser(fs, s)] for s in seeds])
    return out


def common_seeds(counts, c, cell_list, keys_fn):
    seeds = None
    for fs in cell_list:
        have = {s for (cc, f, s), d in counts.items()
                if cc == c and f == fs and all(k in d for k in keys_fn(fs, s))}
        seeds = have if seeds is None else seeds & have
    return sorted(seeds or [])


def summarise_draws(est, draws, deltas=(EQUIV_MARGIN,)):
    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
    lo90, hi90 = np.percentile(draws, [5, 95])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    out = dict(est=est, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90, p_boot=p)
    for dl in deltas:
        out[f'equiv_{dl:g}'] = bool(lo90 > -dl and hi90 < dl)
    return out


def corpus_comparison(counts, c, cell_list, keyA, keyB, metric='f1', stratum='all',
                      two_way=True):
    """keyA/keyB: callables (fs, seed) -> counts key."""
    seeds = common_seeds(counts, c, cell_list, lambda fs, s: [keyA(fs, s), keyB(fs, s)])
    if len(seeds) < 6:
        return None, None
    TA = tensors(counts, c, cell_list, seeds, keyA)
    TB = tensors(counts, c, cell_list, seeds, keyB)
    n_test = next(iter(TA.values())).shape[1]
    boot = Boot(n_test, len(seeds), two_way=two_way)
    est = float(corpus_stat(TA, TB, None, metric, stratum)[0])
    draws = corpus_stat(TA, TB, boot, metric, stratum)
    return dict(n_seeds=len(seeds), **summarise_draws(est, draws, SENS_MARGINS)), draws


def fixed(cfg):
    return lambda fs, s: f'{cfg}__r0'


def corpus_level(counts, R, comparisons, label, metric='f1', stratum='all',
                 cell_filter=None):
    """Per-corpus two-way bootstrap + seed-only sensitivity + descriptive
    nine-cell summary (mean of the corpus draws, corpora resampled
    independently; conditional on these three corpora)."""
    rows = []
    for name, ka, kb in comparisons:
        draws_by_corpus = {}
        for c in CORPUS_ORDER:
            cl = [fs for fs in CELL_ORDER if (c, fs) in cells(R)
                  and (cell_filter is None or fs in cell_filter)]
            if not cl:
                continue
            res, draws = corpus_comparison(counts, c, cl, ka, kb, metric, stratum, True)
            if res is None:
                continue
            res_s, _ = corpus_comparison(counts, c, cl, ka, kb, metric, stratum, False)
            draws_by_corpus[c] = (res['est'], draws)
            rows.append(dict(analysis=label, comparison=name, metric=metric,
                             stratum=stratum, level=c, cells='+'.join(cl), **res,
                             ci95_lo_seedonly=res_s['ci95_lo'],
                             ci95_hi_seedonly=res_s['ci95_hi']))
        if len(draws_by_corpus) >= 2:
            est = np.mean([v[0] for v in draws_by_corpus.values()])
            draws = np.mean([v[1] for v in draws_by_corpus.values()], axis=0)
            rows.append(dict(analysis=label, comparison=name, metric=metric,
                             stratum=stratum, level='ALL (descriptive)',
                             cells=f'{3 * len(draws_by_corpus)} cells',
                             n_seeds=np.nan, **summarise_draws(est, draws, SENS_MARGINS)))
    return pd.DataFrame(rows)


# =============================================================================
# MIXED MODEL (sensitivity)
# =============================================================================
def mixed_model(R, a, b, metric='f1'):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return pd.DataFrame()
    recs = []
    for c, fs in cells(R):
        pv = piv(R, c, fs, metric)
        if a in pv and b in pv:
            x = pv[[a, b]].dropna()
            for s, v in ((x[a] - x[b]) * 100).items():
                recs.append(dict(corpus=c, cell=f'{c}|{fs}', group=f'{c}|{s}', d=v))
    D = pd.DataFrame(recs); rows = []
    if D.empty:
        return D
    subsets = [(c, D[D.corpus == c]) for c in CORPUS_ORDER if (D.corpus == c).any()]
    subsets.append(('ALL (descriptive)', D))
    for lvl, d in subsets:
        try:
            fit = smf.mixedlm('d ~ 0 + C(cell)', d, groups=d['group']).fit(reml=True)
            names = [n for n in fit.fe_params.index]
            L = np.full(len(names), 1.0 / len(names))
            est = float(L @ fit.fe_params[names])
            se = float(np.sqrt(L @ fit.cov_params().loc[names, names].values @ L))
            rows.append(dict(comparison=f'{a} vs {b}', level=lvl, est=est, se=se,
                             ci95_lo=est - 1.96 * se, ci95_hi=est + 1.96 * se,
                             seed_var=float(fit.cov_re.iloc[0, 0]), resid_var=float(fit.scale)))
        except Exception as e:
            rows.append(dict(comparison=f'{a} vs {b}', level=lvl, est=np.nan,
                             note=f'failed: {type(e).__name__}'))
    return pd.DataFrame(rows)


# =============================================================================
# A. INTEGRITY CHECKS
# =============================================================================
def integrity(R, counts, A):
    H('A. INTEGRITY CHECKS')
    R0 = R[R.repeat == 0]
    # expected runs
    exp = R0.groupby(['corpus', 'few_shot', 'config']).seed.nunique().unstack()
    P('runs per cell and configuration (number of seeds):')
    P(exp.fillna(0).astype(int).to_string())

    # step parity
    bad = []
    for r in R0.itertuples():
        want = r.s_aug5 if r.budget == 'match' else int(str(r.budget)[:-1]) * r.s_base5
        if int(r.n_steps) != int(want):
            bad.append((r.corpus, r.few_shot, r.seed, r.config, r.n_steps, want))
    P(f'\nstep parity: {len(bad)} runs deviate from their integer budget'
      + ('' if not bad else f' -> {bad[:5]}'))
    m = R0[R0.budget == 'match']
    g = m.groupby(['corpus', 'few_shot', 'seed']).n_steps.nunique()
    P(f'step-matched configs with identical steps in every (cell, seed): {bool((g == 1).all())}')
    aug = m[m.method.isin(['Dup', 'MR', 'COSINER'])]
    g2 = aug.groupby(['corpus', 'few_shot', 'seed']).n_train.nunique()
    P(f'Dup/MR/COSINER with identical n_train in every (cell, seed): {bool((g2 == 1).all())}')
    g3 = R0.groupby(['corpus', 'few_shot', 'seed', 'method']).train_hash.nunique()
    P(f'identical training set across budgets for each method: {bool((g3 == 1).all())}')
    P(f'runs with skipped fp16 steps: {int((R0.n_skipped > 0).sum())}')
    P(f'runs with unlabelled (truncated) test words: {int((R0.test_words_unlabelled > 0).sum())}')
    sq = pd.to_numeric(R0.seqeval_f1, errors='coerce')
    if sq.notna().any():
        P(f'max |seqeval - internal F1|: {np.nanmax(np.abs(sq - R0.f1)):.2e}')

    # counts consistent with CSV
    diffs = []
    for r in R0.itertuples():
        C = counts.get((r.corpus, r.few_shot, int(r.seed)), {}).get(f'{r.config}__r0')
        if C is not None:
            t = C.sum(0); diffs.append(abs(_metric(t[0], t[1], t[2], 'f1') - r.f1))
    if diffs:
        P(f'per-sentence counts reproduce CSV F1: max |diff| = {max(diffs):.2e} '
          f'over {len(diffs)} runs')

    # stratification
    s = R0.groupby(['corpus', 'few_shot'])[['n_seen', 'n_unseen', 'cov_test']].mean()
    s['prop_seen'] = s.n_seen / (s.n_seen + s.n_unseen)
    s['gap'] = (s.prop_seen - s.cov_test).abs()
    P('\nstratification sanity (prop_seen must equal cov_test):')
    P(s.round(4).to_string())

    # repeatability
    rep = R[R.repeat == 1]
    if not rep.empty:
        mrg = rep.merge(R0, on=['corpus', 'few_shot', 'config', 'seed'], suffixes=('_r1', '_r0'))
        mrg['abs_diff_f1'] = (mrg.f1_r1 - mrg.f1_r0).abs() * 100
        mrg['same_session'] = mrg.session_r1 == mrg.session_r0
        P('\nrun-to-run repeatability (identical configuration and seed, F1 points):')
        P(mrg[['corpus', 'few_shot', 'config', 'seed', 'same_session', 'abs_diff_f1']]
          .round(4).to_string(index=False))
        P(f'max |diff| = {mrg.abs_diff_f1.max():.4f}')
        save(mrg[['corpus', 'few_shot', 'config', 'seed', 'session_r0', 'session_r1',
                  'gpu_r0', 'gpu_r1', 'f1_r0', 'f1_r1', 'abs_diff_f1']], 'table_repeatability.csv')

    # exact step table (Editor 1)
    st = (R0[R0.config.isin(['Base@1x', 'Base@match', 'MR@match'])]
          .pivot_table(index=['corpus', 'few_shot', 'seed'], columns='config',
                       values=['n_steps', 'n_train', 'epochs_eff'], aggfunc='first'))
    st.columns = [f'{a}|{b}' for a, b in st.columns]
    st = st.reset_index()
    first = R0.groupby(['corpus', 'few_shot', 'seed'])[['n_few', 'n_elig', 's_base5',
                                                         's_aug5']].first().reset_index()
    st = first.merge(st, on=['corpus', 'few_shot', 'seed'])
    st['ratio'] = st.s_aug5 / st.s_base5
    save(st, 'table_steps_per_seed.csv')
    summ = st.groupby(['corpus', 'few_shot']).agg(
        n_few=('n_few', 'first'), n_elig_mean=('n_elig', 'mean'),
        S_b5=('s_base5', 'first'), S_a5_min=('s_aug5', 'min'), S_a5_max=('s_aug5', 'max'),
        ratio_min=('ratio', 'min'), ratio_max=('ratio', 'max'),
        base_match_epochs_min=('epochs_eff|Base@match', 'min'),
        base_match_epochs_max=('epochs_eff|Base@match', 'max'))
    P('\ninteger step budgets (full per-seed table: table_steps_per_seed.csv):')
    P(summ.round(3).to_string())
    save(summ.reset_index(), 'table_steps_summary.csv')

    if not A.empty:
        a = A.groupby(['corpus', 'few_shot', 'method']).agg(
            lex=('lex_size', 'mean'), cand_mean=('cand_mean', 'mean'),
            cand_min=('cand_min', 'min'), target=('copies_target', 'mean'),
            made=('copies_made', 'mean'), failed=('failed_attempts', 'sum'),
            sent_short=('sent_short', 'sum'), rep_per_copy=('rep_per_copy', 'mean'),
            frac_repl=('frac_mentions_replaced', 'mean'),
            abs_dlen=('mean_abs_dlen', 'mean')).reset_index()
        a = a[a.method != 'Dup']
        P('\nreplacement statistics (R1.3; means over seeds, failures summed):')
        P(a.round(3).to_string(index=False))
        save(a, 'table_replacement_stats.csv')


# =============================================================================
# B. MAIN RESULTS AND DECOMPOSITION
# =============================================================================
def main_tables(R):
    H('B. MAIN RESULTS (mean ± SD over seeds) AND DECOMPOSITION PER CELL')
    R0 = R[R.repeat == 0]
    g = R0.groupby(['corpus', 'few_shot', 'config']).agg(
        steps=('n_steps', 'mean'), n_train=('n_train', 'mean'),
        epochs=('epochs_eff', 'mean'), f1=('f1', 'mean'), f1_sd=('f1', 'std'),
        p=('precision', 'mean'), p_sd=('precision', 'std'),
        r=('recall', 'mean'), r_sd=('recall', 'std'), n=('f1', 'size')).reset_index()
    for c in ['f1', 'f1_sd', 'p', 'p_sd', 'r', 'r_sd']:
        g[c] *= 100
    save(g, 'table_main_results.csv')
    prim = g[g.config.isin(['Base@1x', 'Base@match', 'Dup@match', 'MR@match',
                            'COSINER@match'])]
    P(prim.round(2).to_string(index=False))

    recs = []
    for c, fs in cells(R):
        pv = piv(R, c, fs)
        row = dict(corpus=c, few_shot=fs)
        for name, a, b in DECOMP:
            if a in pv and b in pv:
                x = pv[[a, b]].dropna(); row[name] = ((x[a] - x[b]) * 100).mean()
        recs.append(row)
    D = pd.DataFrame(recs)
    P('\ndecomposition per cell (F1 points):')
    P(D.round(2).to_string(index=False))
    P('positive cells: ' + ', '.join(f'{k} {int((D[k] > 0).sum())}/{D[k].notna().sum()}'
                                    for k, _, _ in DECOMP if k in D))
    save(D, 'table_decomposition_cells.csv')


# =============================================================================
# C/D. PER-CELL AND PER-CORPUS INFERENCE
# =============================================================================
def inference(R, counts):
    H('C. PER-CELL TESTS (descriptive; Wilcoxon difference + Wilcoxon TOST, '
      f'Holm within family, margin {EQUIV_MARGIN})')
    T1 = per_cell_family(R, PRIMARY, 'primary')
    T2 = per_cell_family(R, SECONDARY, 'secondary (COSINER)')
    T = pd.concat([T1, T2], ignore_index=True)
    if not T.empty:
        cols = ['corpus', 'few_shot', 'family', 'comparison', 'd', 'sd', 'p_diff',
                'p_diff_holm', 'p_tost', 'p_tost_holm', 'verdict', 'p_diff_t', 'p_tost_t']
        P(T[cols].round(4).to_string(index=False))
        save(T, 'table_per_cell_tests.csv')
        P('\nverdict counts:')
        P(T.groupby(['comparison', 'verdict']).size().unstack(fill_value=0).to_string())

    H('D. PER-CORPUS INFERENCE: two-way cluster bootstrap (seeds x test sentences)')
    P(f'B = {N_BOOT}. Equivalence = 90% CI inside (-margin, +margin), i.e. TOST at 0.05.')
    P('Rows "ALL" average the corpora with independent resampling per corpus; they are')
    P('conditional on these three corpora and are reported as DESCRIPTIVE only.')
    comps = [(n, fixed(a), fixed(b)) for n, a, b in DECOMP] + \
            [(f'{a} vs {b}', fixed(a), fixed(b)) for a, b in PRIMARY + SECONDARY
             if (a, b) not in [('MR@match', 'Base@1x')]]
    C = corpus_level(counts, R, comps, 'step-matched')
    if not C.empty:
        show = ['comparison', 'level', 'est', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi',
                'p_boot', 'equiv_0.5', 'equiv_1', 'equiv_1.5',
                'ci95_lo_seedonly', 'ci95_hi_seedonly']
        P(C[[x for x in show if x in C]].round(3).to_string(index=False))
    MM = pd.concat([mixed_model(R, a, b) for _, a, b in DECOMP], ignore_index=True)
    if not MM.empty:
        P('\nsensitivity: linear mixed model, fixed cell effects + random seed intercept')
        P('(averages the cell effects; accounts for nesting, not for the shared test set)')
        P(MM.round(3).to_string(index=False))
        save(MM, 'table_mixed_model.csv')

    H('D2. STRATIFIED BY MENTION VISIBILITY (per corpus, two-way bootstrap)')
    S = pd.concat([corpus_level(counts, R, [(n, fixed(a), fixed(b))
                                            for n, a, b in DECOMP[:3]],
                                'step-matched', 'f1', st) for st in ('seen', 'unseen')],
                  ignore_index=True)
    if not S.empty:
        P(S[['comparison', 'stratum', 'level', 'est', 'ci95_lo', 'ci95_hi']]
          .round(3).to_string(index=False))

    H('D3. PRECISION-RECALL SHIFT at identical steps (vs Base@match)')
    recs = []
    for c, fs in cells(R):
        for m in ['MR@match', 'COSINER@match']:
            row = dict(corpus=c, few_shot=fs, method=m)
            for met in ['precision', 'recall', 'f1']:
                pv = piv(R, c, fs, met)
                if m in pv and 'Base@match' in pv:
                    x = pv[[m, 'Base@match']].dropna()
                    row['d_' + met] = ((x[m] - x['Base@match']) * 100).mean()
            recs.append(row)
    PR = pd.DataFrame(recs)
    if not PR.empty and 'd_precision' in PR:
        P(PR.round(2).to_string(index=False))
        for m in PR.method.unique():
            x = PR[PR.method == m]
            exc_p = x[x.d_precision >= 0][['corpus', 'few_shot']].values.tolist()
            exc_r = x[x.d_recall <= 0][['corpus', 'few_shot']].values.tolist()
            P(f'{m}: precision lower in {int((x.d_precision < 0).sum())}/{len(x)} cells '
              f'(exceptions {exc_p}); recall higher in {int((x.d_recall > 0).sum())}/{len(x)} '
              f'cells (exceptions {exc_r})')
        save(PR, 'table_pr_shift_cells.csv')
        PRc = pd.concat([corpus_level(counts, R, [(f'{m} vs Base@match', fixed(m),
                                                   fixed('Base@match'))],
                                      'pr-shift', met)
                         for m in ['MR@match', 'COSINER@match']
                         for met in ['precision', 'recall']], ignore_index=True)
        if not PRc.empty:
            P(PRc[['comparison', 'metric', 'level', 'est', 'ci95_lo', 'ci95_hi']]
              .round(3).to_string(index=False))
            C = pd.concat([C, S, PRc], ignore_index=True)
    save(C, 'table_corpus_level.csv')
    return T, C


# =============================================================================
# E. TRAINING-BUDGET SWEEP (Reviewer 1, comment 1)
# =============================================================================
def sweep(R, counts):
    H('E. TRAINING-BUDGET SWEEP: identical integer step budgets for every method')
    R0 = R[(R.repeat == 0) & R.method.isin(SWEEP_METHODS + ['Dup'])]
    nb = (R0[R0.method.isin(SWEEP_METHODS)]
          .groupby(['corpus', 'few_shot', 'method']).budget.nunique().unstack())
    sc = [(c, fs) for c, fs in cells(R) if (c, fs) in nb.index
          and nb.loc[(c, fs)].reindex(SWEEP_METHODS).min() >= 3]
    if not sc:
        P('no cell has a full budget sweep yet'); return None, None
    P('cells with a full budget sweep: ' + ', '.join(f'{c} {fs}' for c, fs in sc))
    P('(cells without a sweep keep only the step-matched comparison; see B-D)')
    R0 = R0[[(c, fs) in sc for c, fs in zip(R0.corpus, R0.few_shot)]]
    g = R0.groupby(['corpus', 'few_shot', 'method', 'budget']).agg(
        steps=('n_steps', 'mean'), mult=('budget_mult', 'mean'), f1=('f1', 'mean'),
        sd=('f1', 'std'), n=('f1', 'size'), dev_f1=('dev_f1', 'mean')).reset_index()
    g['ci95'] = g.apply(lambda r: stats.t.ppf(0.975, r.n - 1) * r.sd / np.sqrt(r.n)
                        if r.n > 1 else np.nan, axis=1)
    for c in ['f1', 'sd', 'ci95', 'dev_f1']:
        g[c] *= 100
    g = g.sort_values(['corpus', 'few_shot', 'method', 'steps'])
    save(g, 'table_sweep_curves.csv')
    P(g[g.method != 'Dup'].round(2).to_string(index=False))

    # per-budget paired differences
    recs = []
    for c, fs in sc:
        pv = piv(R, c, fs)
        for m in ['MR', 'COSINER']:
            for b in sorted({x.split('@')[1] for x in pv.columns if x.startswith(m + '@')}):
                a_, b_ = f'{m}@{b}', f'Base@{b}'
                if b_ not in pv:
                    continue
                x = pv[[a_, b_]].dropna()
                if len(x) < 6:
                    continue
                t = paired_tests((x[a_] - x[b_]).values * 100)
                recs.append(dict(corpus=c, few_shot=fs, comparison=f'{m} - Base',
                                 budget=b, steps=R0[(R0.corpus == c) & (R0.few_shot == fs) &
                                                    (R0.config == b_)].n_steps.mean(), **t))
    PB = pd.DataFrame(recs)
    if not PB.empty:
        PB = PB.sort_values(['corpus', 'few_shot', 'comparison', 'steps'])
        PB['p_diff_holm'] = PB.groupby(['corpus', 'few_shot', 'comparison']).p_diff \
                              .transform(lambda p: holm_adjust(p.values))
        P('\npaired difference at each identical budget (Holm across budgets within cell):')
        P(PB[['corpus', 'few_shot', 'comparison', 'budget', 'steps', 'd', 'ci95_lo',
              'ci95_hi', 'p_diff', 'p_diff_holm']].round(3).to_string(index=False))
        save(PB, 'table_sweep_per_budget.csv')

    # still rising at the largest budget?
    recs = []
    for c, fs in sc:
        pv = piv(R, c, fs)
        for m in SWEEP_METHODS:
            sub = g[(g.corpus == c) & (g.few_shot == fs) & (g.method == m)].sort_values('steps')
            if len(sub) < 2:
                continue
            last, prev = f'{m}@{sub.budget.iloc[-1]}', f'{m}@{sub.budget.iloc[-2]}'
            x = pv[[last, prev]].dropna()
            t = paired_tests((x[last] - x[prev]).values * 100)
            recs.append(dict(corpus=c, few_shot=fs, method=m, last=last, previous=prev,
                             d=t['d'], ci95_lo=t['ci95_lo'], ci95_hi=t['ci95_hi'],
                             p_diff=t['p_diff'],
                             peak_budget=sub.loc[sub.f1.idxmax(), 'budget']))
    TR = pd.DataFrame(recs)
    if not TR.empty:
        P('\ncurve shape: change between the two largest budgets, and the budget with '
          'the highest mean test F1')
        P(TR.round(3).to_string(index=False))
        save(TR, 'table_sweep_curve_shape.csv')

    # best of curve: dev-selected (practical) and oracle (exploratory)
    H('E2. BEST OF CURVE: budget selected on the few-shot dev set (practical)')
    sel = {}
    recs = []
    for (c, fs, s, m), sub in R0[R0.method.isin(SWEEP_METHODS)].groupby(
            ['corpus', 'few_shot', 'seed', 'method']):
        best = sub.sort_values(['dev_f1', 'n_steps'], ascending=[False, True]).iloc[0]
        sel[(c, fs, s, m)] = best.config
        recs.append(dict(corpus=c, few_shot=fs, seed=s, method=m, chosen=best.budget,
                         steps=best.n_steps, f1=best.f1 * 100, dev_f1=best.dev_f1 * 100))
    SEL = pd.DataFrame(recs)
    save(SEL, 'table_dev_selection_per_seed.csv')
    if SEL.empty:
        return g, None
    P('chosen budgets (count over seeds):')
    P(SEL.groupby(['corpus', 'few_shot', 'method']).chosen.value_counts()
      .unstack(fill_value=0).to_string())
    recs = []
    for (c, fs), sub in SEL.groupby(['corpus', 'few_shot']):
        pv = sub.pivot_table(index='seed', columns='method', values='f1')
        for m in ['MR', 'COSINER']:
            if m in pv and 'Base' in pv:
                x = pv[[m, 'Base']].dropna()
                if len(x) >= 6:
                    recs.append(dict(corpus=c, few_shot=fs, comparison=f'{m}* - Base*',
                                     f1_method=x[m].mean(), f1_base=x['Base'].mean(),
                                     **paired_tests((x[m] - x['Base']).values)))
    DS = pd.DataFrame(recs)
    if not DS.empty:
        P('\nper cell (descriptive):')
        P(DS[['corpus', 'few_shot', 'comparison', 'f1_method', 'f1_base', 'd', 'ci95_lo',
              'ci95_hi', 'p_diff', 'p_tost', 'pos']].round(3).to_string(index=False))
        save(DS, 'table_dev_selected_cells.csv')

    def chooser(m):
        return lambda fs, s: f"{sel[(c_cur[0], fs, s, m)]}__r0"
    rows = []
    for c in CORPUS_ORDER:
        c_cur = [c]
        cl = [fs for fs in CELL_ORDER if (c, fs) in sc]
        if not cl:
            continue
        for m in ['MR', 'COSINER']:
            try:
                res, _ = corpus_comparison(counts, c, cl, chooser(m), chooser('Base'))
            except KeyError:
                res = None
            if res:
                rows.append(dict(comparison=f'{m}* - Base* (dev-selected)', level=c, **res))
    CS = pd.DataFrame(rows)
    if not CS.empty:
        P('\nper corpus, two-way cluster bootstrap:')
        P(CS[['comparison', 'level', 'est', 'ci95_lo', 'ci95_hi', 'ci90_lo', 'ci90_hi',
              'p_boot', 'equiv_1']].round(3).to_string(index=False))
        save(CS, 'table_dev_selected_corpus.csv')

    H('E3. BEST OF CURVE: test-set peak (ORACLE — exploratory, not a usable procedure)')
    recs = []
    for (c, fs), sub in g[g.method.isin(SWEEP_METHODS)].groupby(['corpus', 'few_shot']):
        row = dict(corpus=c, few_shot=fs)
        for m in SWEEP_METHODS:
            s = sub[sub.method == m]
            if len(s):
                k = s.f1.idxmax()
                row[f'{m}_peak'] = s.loc[k, 'f1']; row[f'{m}_at'] = s.loc[k, 'budget']
        if 'MR_peak' in row and 'Base_peak' in row:
            row['MR-Base'] = row['MR_peak'] - row['Base_peak']
        if 'COSINER_peak' in row and 'Base_peak' in row:
            row['COSINER-Base'] = row['COSINER_peak'] - row['Base_peak']
        recs.append(row)
    OR = pd.DataFrame(recs)
    P(OR.round(2).to_string(index=False))
    save(OR, 'table_oracle_peaks.csv')
    return g, SEL


# =============================================================================
# F. FIGURES
# =============================================================================
def figures(R, curves, T, C):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        P('matplotlib not available; figures skipped'); return
    H('F. FIGURES')
    colours = {'Base': '#222222', 'MR': '#d95f02', 'COSINER': '#1b9e77', 'Dup': '#7570b3'}
    panels = ([] if curves is None else
              [(c, fs) for c in CORPUS_ORDER for fs in CELL_ORDER
               if ((curves.corpus == c) & (curves.few_shot == fs)).any()])
    if panels:
        nc = min(3, len(panels)); nr = int(np.ceil(len(panels) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.6 * nr), squeeze=False)
        for ax in axes.flat[len(panels):]:
            ax.axis('off')
        for k, (c, fs) in enumerate(panels):
                i, j = divmod(k, nc)
                ax = axes[i, j]
                sub = curves[(curves.corpus == c) & (curves.few_shot == fs)]
                if sub.empty:
                    ax.axis('off'); continue
                for m in SWEEP_METHODS:
                    s = sub[sub.method == m].sort_values('steps')
                    if s.empty:
                        continue
                    jit = {'Base': 1.0, 'MR': 1.04, 'COSINER': 0.96}[m]  # visual offset only
                    ax.errorbar(s.steps * jit, s.f1, yerr=s.ci95, marker='o', ms=4, capsize=3,
                                lw=1.4, color=colours[m], label=m,
                                ls='-' if m == 'Base' else '--')
                s = sub[(sub.method == 'Dup') & (sub.budget == 'match')]
                if len(s):
                    ax.errorbar(s.steps, s.f1, yerr=s.ci95, marker='s', ms=6, capsize=3,
                                color=colours['Dup'], ls='none', label='Dup (step-matched)')
                ax.set_xscale('log', base=2)
                ax.set_title(f'{c} {fs}', fontsize=10)
                if i == nr - 1: ax.set_xlabel('optimizer steps (log scale)')
                if j == 0: ax.set_ylabel('test F1')
                ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle('F1 versus training budget. Points: mean over seeds; error bars: 95% '
                     't-confidence interval over seeds; MR/COSINER offset horizontally by '
                     '4% for legibility (all methods share identical step budgets)', fontsize=9)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'fig_step_curves.{ext}'), dpi=300)
        plt.close(fig); P('saved fig_step_curves.png/.pdf')

    if T is not None and not T.empty:
        comps = ['MR@match vs Base@match', 'MR@match vs Dup@match',
                 'COSINER@match vs Base@match']
        fig, axes = plt.subplots(1, len(comps), figsize=(13, 5), sharey=True)
        for k, comp in enumerate(comps):
            ax = axes[k]; labels, y = [], 0
            for c in CORPUS_ORDER:
                for fs in CELL_ORDER:
                    r = T[(T.comparison == comp) & (T.corpus == c) & (T.few_shot == fs)]
                    if len(r):
                        r = r.iloc[0]
                        ax.errorbar(r.d, y, xerr=[[r.d - r.ci90_lo], [r.ci90_hi - r.d]],
                                    fmt='o', color='#555555', capsize=3, ms=4)
                        labels.append(f'{c} {fs}'); y += 1
                if C is not None and not C.empty:
                    q = C[(C.comparison == comp) & (C.level == c) & (C.metric == 'f1')
                          & (C.stratum == 'all')]
                    if len(q):
                        q = q.iloc[0]
                        ax.errorbar(q.est, y, xerr=[[q.est - q.ci90_lo], [q.ci90_hi - q.est]],
                                    fmt='D', color='#d95f02', capsize=4, ms=6)
                        labels.append(f'{c} (corpus)'); y += 1
            ax.axvspan(-EQUIV_MARGIN, EQUIV_MARGIN, color='#1b9e77', alpha=0.12)
            ax.axvline(0, color='k', lw=0.8)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis(); ax.set_title(comp, fontsize=10)
            ax.set_xlabel('difference in F1 points')
        fig.suptitle('Equivalence. Grey: per cell, mean and 90% t-CI over seeds. Orange: per '
                     'corpus, 90% two-way cluster-bootstrap CI. Band: margin of '
                     f'{EQUIV_MARGIN:g} F1 point', fontsize=10)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'fig_equivalence.{ext}'), dpi=300)
        plt.close(fig); P('saved fig_equivalence.png/.pdf')


# =============================================================================
def main():
    R, counts, A = load()
    integrity(R, counts, A)
    main_tables(R)
    T, C = inference(R, counts)
    curves, _ = sweep(R, counts)
    figures(R, curves, T, C)
    with open(os.path.join(OUT, 'report.txt'), 'w') as f:
        f.write('\n'.join(_LOG))
    print(f'\nall tables, figures and report.txt written to {OUT}')


if __name__ == '__main__':
    main()
