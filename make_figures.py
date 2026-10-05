# =============================================================================
#  make_figures.py — figures of the manuscript that are not produced by
#  analysis_steps_v2.py (Figures 6 and 7 are produced there).
#
#    Figure 1  evaluation protocol (diagram)
#    Figure 2  step ratio S_aug / S_base per cell and seed
#    Figure 3  control protocol (diagram)
#    Figure 4  main results (five primary configurations)
#    Figure 5  decomposition of the gain (optimisation / content)
#    Figure 8  precision-recall shift relative to Base-match
#    Figure 9  decomposition by mention visibility (seen / unseen)
#
#  Usage:  python make_figures.py --results outputs --analysis analysis --out figures
#  Reads every steps_*/results.csv below --results, and table_steps_per_seed.csv
#  and table_pr_shift_cells.csv from --analysis. CPU only.
# =============================================================================
import argparse, glob, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy import stats

CODE_VERSION = '2.1'
CORPORA = [('NCBI-Disease', 'NCBI-Disease'), ('BC5CDR-Chem', 'BC5CDR-Chemical'), ('BC2GM', 'BC2GM')]
CELLS = ['2%', '5%', '10%']
PRIMARY = ['Base@1x', 'Base@match', 'Dup@match', 'MR@match', 'COSINER@match']
NAMES = ['Base-E5', 'Base-match', 'Duplicate', 'MR', 'COSINER']
COLOURS = ['#cccccc', '#4d4d4d', '#7570b3', '#d95f02', '#1b9e77']
COMPONENTS = [('Optimisation', 'Base@match', 'Base@1x', '#4d4d4d'),
              ('Content (MR)', 'MR@match', 'Base@match', '#d95f02'),
              ('Content (COSINER)', 'COSINER@match', 'Base@match', '#1b9e77')]


def load_results(root):
    files = [f for f in glob.glob(os.path.join(root, '**', 'results.csv'), recursive=True)
             if re.search(r'steps_(ncbi|bc5cdr|bc2gm)', f)]
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    R = R[(R.code_version.astype(str) == CODE_VERSION) & (R.repeat == 0)]
    return R.drop_duplicates(['corpus', 'few_shot', 'config', 'seed'], keep='last')


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def style(ax):
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def cell_axis(ax, y=-0.14):
    cells = [(c, f) for c, _ in CORPORA for f in CELLS]
    ax.set_xticks(range(9)); ax.set_xticklabels([f for _, f in cells])
    for b in (2.5, 5.5):
        ax.axvline(b, color='#bbbbbb', lw=0.8, ls='--')
    for i, (_, lab) in enumerate(CORPORA):
        ax.text(3 * i + 1, y, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=10)
    ax.set_xlim(-0.6, 8.6)


def paired(R, c, f, a, b, col='f1'):
    p = R[(R.corpus == c) & (R.few_shot == f)].pivot_table(index='seed', columns='config', values=col)
    d = (p[a] - p[b]).dropna().values * 100
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


# ----------------------------------------------------------------- diagrams --
def _box(ax, x, y, w, h, fc, ec, lines, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec=ec, lw=1.3))
    n = len(lines)
    for i, (t, kw) in enumerate(lines):
        ax.text(x + w / 2, y + h - (i + 0.75) * h / (n + 0.5), t, ha='center', va='center',
                fontsize=kw.get('fs', fs), weight=kw.get('w', 'normal'), style=kw.get('st', 'normal'))


def _arrow(ax, x1, y1, x2, y2, **kw):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=1.1, **kw))


def figure1(out):
    B = dict
    fig, ax = plt.subplots(figsize=(13, 10.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 10.2); ax.axis('off')
    def num(y, k):
        ax.add_patch(Circle((0.45, y), 0.27, color='#0b2c5a'))
        ax.text(0.45, y, str(k), ha='center', va='center', color='white', fontsize=13, weight='bold')
    _box(ax, 1, 9.35, 11.8, 0.7, '#e3eefb', '#5b7fb5', [('Benchmark corpora:  NCBI-Disease (disease)  •  BC5CDR-Chemical (chemical)  •  BC2GM (gene)', B(w='bold'))]); num(9.7, 1)
    _arrow(ax, 6.9, 9.35, 6.9, 9.0)
    _box(ax, 1, 8.15, 11.8, 0.85, '#e8f5e9', '#5a9b62', [('Few-shot sampling:  r ∈ {2%, 5%, 10%}  •  10 paired seeds per cell', B(w='bold')), ('a new subset is drawn for every seed; subsets of a seed are nested across ratios', B(fs=10.3))]); num(8.57, 2)
    ax.text(6.9, 7.92, 'all configurations share the same few-shot subset within a seed', ha='center', fontsize=10, style='italic', color='#1f2f5a')
    ax.plot([2.5, 10.725], [7.75, 7.75], color='black', lw=1.1); ax.plot([6.9, 6.9], [7.85, 7.75], color='black', lw=1.1)
    for x in (2.5, 6.325, 10.725):
        _arrow(ax, x, 7.75, x, 7.45)
    _box(ax, 1, 5.65, 3.0, 1.8, '#f2f2f2', '#555555', [('Reference', B(w='bold', fs=12)), ('Base-E5', B()), ('original data, 5 epochs', B()), (r'$S_{\mathrm{base}}=\lceil n/B\rceil\cdot 5$ steps', B())])
    _box(ax, 4.25, 5.65, 4.15, 1.8, '#e3eefb', '#3d6fb6', [(r'Controls  ($S_{\mathrm{aug}}$ steps)', B(w='bold', fs=12)), ('Base-match: original data only', B()), ('Duplicate: K exact copies of each', B()), ('mention-bearing sentence', B()), ('no new information', B(st='italic'))])
    _box(ax, 8.65, 5.65, 4.15, 1.8, '#fbe3e5', '#c04a57', [(r'Augmented  ($S_{\mathrm{aug}}$ steps)', B(w='bold', fs=12)), ('MR: random candidate order', B()), ('COSINER: similarity ranking', B()), ('K copies per mention-bearing', B()), ('sentence, every mention replaced', B())]); num(6.55, 3)
    ax.text(6.9, 5.35, r'Training-budget sweep (2% cells only): Base, MR and COSINER at 1×, 2×, 8× and 16× $S_{\mathrm{base}}$ and at $S_{\mathrm{aug}}$; budget selected on a few-shot development set', ha='center', fontsize=9.3, color='#333333')
    _arrow(ax, 6.9, 5.2, 6.9, 4.95)
    _box(ax, 1, 4.05, 11.8, 0.9, '#fdf3dc', '#c9a227', [('Fine-tuning BioBERT v1.1  •  AdamW, LR 5×10⁻⁵ decaying linearly to zero  •  batch size 8', B(w='bold', fs=10.3)), ('exact integer step budget, stopping mid-epoch when required  •  no checkpoint selection  •  deterministic', B(fs=10.3))]); num(4.5, 4)
    _arrow(ax, 4.0, 4.05, 4.0, 3.8); _arrow(ax, 9.9, 4.05, 9.9, 3.8)
    _box(ax, 1, 2.95, 5.8, 0.85, '#e8f5e9', '#5a9b62', [('Entity-level evaluation', B(w='bold', fs=11)), ('precision • recall • F1, exact span and type match', B())])
    _box(ax, 7.0, 2.95, 5.8, 0.85, '#e8f5e9', '#5a9b62', [('Stratified evaluation', B(w='bold', fs=11)), ('seen versus unseen mention surface forms', B())]); num(3.37, 5)
    _arrow(ax, 6.9, 2.95, 6.9, 2.7)
    _box(ax, 1, 1.65, 11.8, 1.05, '#e9ecfb', '#5966b8', [('Statistical analysis:  per corpus, two-way cluster bootstrap over seeds and test sentences', B(w='bold', fs=10.3)), ('Holm-adjusted difference tests  •  equivalence within δ = 1.0 F1 point (sensitivity 0.5 / 1.5)', B(fs=10.3)), ('per cell (descriptive): Wilcoxon and Wilcoxon-TOST, Holm within cell', B(fs=10.3))]); num(2.17, 6)
    _arrow(ax, 6.9, 1.65, 6.9, 1.4)
    _box(ax, 2.1, 0.55, 9.6, 0.85, '#dff1df', '#2e7d32', [('Decomposition of the reported gain:', B(w='bold', fs=11)), ('OPTIMISATION = Base-match − Base-E5        CONTENT = augmented − Base-match', B(w='bold'))])
    ax.text(6.9, 0.15, r'$K=5$ copies;  $n$ = few-shot subset size;  $B=8$;  $S_{\mathrm{aug}}=\lceil (n+K\,n_{\mathrm{elig}})/B\rceil\cdot 5$, where $n_{\mathrm{elig}}$ is the number of mention-bearing sentences', ha='center', fontsize=9, color='#333333')
    save(fig, out, 'fig1_protocol')


def figure3(out):
    B = dict
    fig, ax = plt.subplots(figsize=(13, 6.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis('off')
    ax.text(6.5, 6.35, 'Control protocol: separating the optimisation component from the content component', ha='center', fontsize=13, weight='bold')
    _box(ax, 4.6, 5.25, 3.8, 0.8, '#e3eefb', '#3d6fb6', [(r'Few-shot subset $D_r$', B(w='bold', fs=11)), ('(original sentences)', B(w='bold'))])
    xs, w = [0.3, 3.5, 6.7, 9.9], 2.9
    cfg = [('#f2f2f2', '#555555', [('Base-E5', B(w='bold', fs=11)), ('original data', B()), ('5 epochs', B()), (r'$S_0$ steps', B())]),
           ('#dddddd', '#555555', [('Base-match', B(w='bold', fs=11)), ('original data', B()), ('stopped at step S', B()), (r'$S$ steps', B())]),
           ('#cfe0f5', '#3d6fb6', [('Duplicate', B(w='bold', fs=11)), ('K exact copies of each', B()), ('mention-bearing sentence', B()), (r'5 epochs, $S$ steps', B())]),
           ('#fbe3e5', '#c04a57', [('MR / COSINER', B(w='bold', fs=11)), ('K copies of each mention-', B()), ('bearing sentence, every', B()), ('mention replaced', B()), (r'5 epochs, $S$ steps', B())])]
    for x, (fc, ec, l) in zip(xs, cfg):
        _box(ax, x, 3.0, w, 1.75, fc, ec, l)
        _arrow(ax, 6.5, 5.25, x + w / 2, 4.75); _arrow(ax, x + w / 2, 3.0, x + w / 2, 2.35)
    _box(ax, 0.3, 1.6, 9.3, 0.75, '#f2f2f2', '#888888', [('No new information (Base-E5, Base-match, Duplicate)', B())])
    _box(ax, 9.9, 1.6, 2.9, 0.75, '#fbe3e5', '#c04a57', [('New information added', B())])
    def span(x1, x2, y, txt, c):
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle='<|-|>', color=c, lw=1.8))
        ax.text((x1 + x2) / 2, y + 0.13, txt, ha='center', va='bottom', color=c, fontsize=10.5, weight='bold')
    span(xs[0] + w / 2, xs[1] + w / 2, 1.15, r'OPTIMISATION: same data, $S_0$ vs $S$ steps', '#1565c0')
    span(xs[1] + w / 2, xs[3] + w / 2, 0.45, 'CONTENT: identical steps, new versus no new information', '#c62828')
    ax.text(6.5, 0.05, r'K = 5.  $S_0=\lceil |D_r|/B\rceil\cdot 5$;  $S=\lceil(|D_r|+|D_{aug}|)/B\rceil\cdot 5$.  Duplicate versus Base-match is the consistency check between the two controls.', ha='center', fontsize=9, color='#333333')
    save(fig, out, 'fig3_controls')


# ------------------------------------------------------------- data figures --
def figure2(steps, out):
    fig, ax = plt.subplots(figsize=(9, 3.8)); rng = np.random.default_rng(0)
    for i, (c, f) in enumerate([(c, f) for c, _ in CORPORA for f in CELLS]):
        r = steps[(steps.corpus == c) & (steps.few_shot == f)].ratio.values
        ax.bar(i, r.mean(), 0.6, color='#9ecae1', edgecolor='#3182bd')
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(r)), r, s=14, color='#08519c', zorder=3)
        ax.text(i, r.max() + 0.06, f'{r.mean():.2f}', ha='center', fontsize=8)
    ax.axhline(1, color='k', lw=0.8, ls='--'); ax.text(8.45, 1.05, 'Base-E5', fontsize=8, ha='right')
    cell_axis(ax); ax.set_ylabel(r'Step ratio $S_{aug}/S_{base}$'); ax.set_ylim(0, 4.7); style(ax)
    save(fig, out, 'fig2_step_ratio')


def figure4(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    limits = {'NCBI-Disease': (62, 86), 'BC5CDR-Chem': (82, 93), 'BC2GM': (62, 80)}
    for ax, (c, lab) in zip(axes, CORPORA):
        for k, (cfg, nm, col) in enumerate(zip(PRIMARY, NAMES, COLOURS)):
            s = [R[(R.corpus == c) & (R.few_shot == f) & (R.config == cfg)].f1 * 100 for f in CELLS]
            ax.bar(np.arange(3) + (k - 2) * 0.16, [x.mean() for x in s], 0.16, yerr=[x.std() for x in s],
                   capsize=2, color=col, label=nm, edgecolor='white', linewidth=0.4, error_kw=dict(lw=0.7))
        ax.set_ylim(*limits[c]); ax.set_xticks(range(3)); ax.set_xticklabels(CELLS); ax.set_title(lab, fontsize=11); style(ax)
    axes[0].set_ylabel('Test F1'); axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout(); save(fig, out, 'fig4_main_results')


def figure5(R, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cells = [(c, f) for c, _ in CORPORA for f in CELLS]
    for k, (nm, a, b, col) in enumerate(COMPONENTS):
        m, h = zip(*[paired(R, c, f, a, b) for c, f in cells])
        ax.bar(np.arange(9) + (k - 1) * 0.26, m, 0.26, yerr=h, capsize=2.5, color=col, label=nm,
               edgecolor='white', linewidth=0.5, error_kw=dict(lw=0.8))
    ax.axhline(0, color='k', lw=0.8); cell_axis(ax, -0.13); ax.set_ylabel('Δ F1 (points)')
    ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=9); style(ax)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.2); save(fig, out, 'fig5_decomposition')


def figure8(pr, out):
    col = {'NCBI-Disease': '#1b9e77', 'BC5CDR-Chem': '#7570b3', 'BC2GM': '#d95f02'}
    lab = dict(CORPORA)
    off = {('MR@match', 'BC2GM', '5%'): (-22, 4), ('MR@match', 'BC5CDR-Chem', '5%'): (-6, -12)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, (m, t) in zip(axes, [('MR@match', 'MR − Base-match'), ('COSINER@match', 'COSINER − Base-match')]):
        ax.axhline(0, color='k', lw=0.8); ax.axvline(0, color='k', lw=0.8)
        ax.fill_between([-3.2, 0], 0, 4.3, color='#eeeeee', zorder=0)
        for r in pr[pr.method == m].itertuples():
            ax.scatter(r.d_precision, r.d_recall, s=55, color=col[r.corpus], edgecolor='k', lw=0.5, zorder=3, label=lab[r.corpus])
            ax.annotate(r.few_shot, (r.d_precision, r.d_recall), xytext=off.get((m, r.corpus, r.few_shot), (5, 4)), textcoords='offset points', fontsize=8)
        ax.set_title(t, fontsize=11); ax.set_xlabel('Δ precision (points)'); ax.set_xlim(-3.2, 0.8); ax.set_ylim(-0.7, 4.3)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Δ recall (points)')
    h, l = axes[0].get_legend_handles_labels(); d = dict(zip(l, h))
    axes[1].legend(d.values(), d.keys(), frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout(); save(fig, out, 'fig8_pr_shift')


def figure9(R, out):
    cells = [(c, f) for c, _ in CORPORA for f in CELLS]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for ax, stratum in zip(axes, ('seen', 'unseen')):
        for k, (nm, a, b, col) in enumerate(COMPONENTS):
            m, h = zip(*[paired(R, c, f, a, b, 'f1_' + stratum) for c, f in cells])
            ax.bar(np.arange(9) + (k - 1) * 0.26, m, 0.26, yerr=h, capsize=2, color=col, label=nm,
                   edgecolor='white', linewidth=0.4, error_kw=dict(lw=0.7))
        ax.axhline(0, color='k', lw=0.8); ax.set_ylabel(f'Δ F1, {stratum} mentions'); style(ax)
        for b in (2.5, 5.5):
            ax.axvline(b, color='#bbbbbb', lw=0.8, ls='--')
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc='upper right'); cell_axis(axes[1], -0.16)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.12); save(fig, out, 'fig9_stratified')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='outputs', help='folder containing steps_<corpus>/results.csv')
    ap.add_argument('--analysis', default='analysis', help='output folder of analysis_steps_v2.py')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    figure1(a.out); figure3(a.out)
    R = load_results(a.results)
    figure2(pd.read_csv(os.path.join(a.analysis, 'table_steps_per_seed.csv')), a.out)
    figure4(R, a.out); figure5(R, a.out)
    figure8(pd.read_csv(os.path.join(a.analysis, 'table_pr_shift_cells.csv')), a.out)
    figure9(R, a.out)


if __name__ == '__main__':
    main()
