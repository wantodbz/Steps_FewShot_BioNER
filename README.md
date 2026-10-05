# More Steps, Not More Data: Re-Examining Data Augmentation for Few-Shot Biomedical NER

Code, data and results for the article

> Agus Siswanto, Bambang Tutuko, Firdaus and Jasmir. *More Steps, Not More Data: Re-Examining Data Augmentation for Few-Shot Biomedical Named Entity Recognition.* Computers, Materials & Continua (under review).

## Overview

Data augmentation enlarges the training set. When an augmented model and a non-augmented baseline are trained for the same number of epochs, the augmented model therefore also receives more gradient steps. This repository separates the two effects using two control configurations:

- **Base-match** trains the non-augmented baseline for exactly the same number of gradient steps as the augmented model.
- **Duplicate** copies each mention-bearing sentence *K* times without changing it, giving the same data volume and the same number of steps as augmentation but no new information.

The augmentation methods evaluated are **mention replacement (MR)** and **COSINER**. Both use the same replacement operator and candidate set. COSINER ranks the candidate mentions by contextual similarity, and MR orders them randomly.

The setting is as follows:
- **Encoder:** BioBERT v1.1.
- **Corpora:** NCBI-Disease, BC5CDR-Chemical and BC2GM.
- **Few-shot ratios:** 2%, 5% and 10% of the training set.
- **Seeds:** ten paired seeds per cell.

The gain of augmentation is decomposed into two components:

```
optimisation component = F1(Base-match) − F1(Base-E5)
content component      = F1(MR or COSINER) − F1(Base-match)
```

Base-E5 is the reference protocol: five epochs on the original data.

## Repository structure

```
Steps_FewShot_BioNER/
├── README.md
├── requirements.txt
├── steps_ncbi.py            # experiments, NCBI-Disease
├── steps_bc5cdr.py          # experiments, BC5CDR-Chemical
├── steps_bc2gm.py           # experiments, BC2GM
├── analysis_steps_v2.py     # statistical analysis, Tables 2–3 and 7–12, Figures 6–7
├── make_figures.py          # Figures 1–5, 8 and 9
├── data/                    # exact benchmark files used (see "Data")
│   ├── ncbi-disease/        train.csv  validation.csv  test.csv
│   ├── bc5cdr/              train.csv  validation.csv  test.csv
│   └── bc2gm-corpus/        train.csv  validation.csv  test.csv
├── outputs/                 # raw experiment outputs (see "Outputs")
│   ├── steps_ncbi/
│   ├── steps_bc5cdr/
│   └── steps_bc2gm/
├── analysis/                # output of analysis_steps_v2.py
└── figures/                 # output of make_figures.py
```

The three experiment scripts are identical except for the `CORPUS` setting, so they can run in parallel on separate machines or accounts.

## Data

The three corpora were obtained from the Hugging Face Hub with their original training, validation and test splits. The exact files used are included in `data/`, and their MD5 checksums are recorded automatically in every experimental session (`outputs/steps_*/meta.json`).

| Corpus | Entity type | Hugging Face dataset | Train / Validation / Test (rows) |
|---|---|---|---|
| NCBI-Disease | Disease | `ncbi_disease` | 5,433 / 924 / 941 |
| BC5CDR | Chemical (disease labels mapped to O) | `tner/bc5cdr` | 5,228 / 5,330 / 5,865 |
| BC2GM | Gene | `bc2gm_corpus` | 12,500 / 2,500 / 5,000 |

| File | MD5 |
|---|---|
| ncbi-disease/train.csv | `9478cf21fb677bf49ee7654dd38643a2` |
| ncbi-disease/validation.csv | `38bb954b3433892bb6ed04c08c48b67c` |
| ncbi-disease/test.csv | `d317d4fca3309df15683c8ac7657a893` |
| bc5cdr/train.csv | `25842c1c3731c6bc5c384b057580b698` |
| bc5cdr/validation.csv | `001214ffbc07427ef3ce4745f7b20952` |
| bc5cdr/test.csv | `bdee78255cc327506e26593cfaa34590` |
| bc2gm-corpus/train.csv | `342d73d895cc7b2fe221d93fc90e9b17` |
| bc2gm-corpus/validation.csv | `edafb85f5fe3ee4080363eb171236579` |
| bc2gm-corpus/test.csv | `7caaa93fc44b8845a9c61b229cf93b95` |

Each NCBI-Disease file ends with one empty row, which the parser removes. No other rows are removed. Please cite the original corpora when using them:

- NCBI-Disease: Doğan et al., *J Biomed Inform* 2014.
- BC5CDR: Li et al., *Database* 2016.
- BC2GM: Smith et al., *Genome Biol* 2008.

## Experimental protocol

| Item | Setting |
|---|---|
| Encoder | `dmis-lab/biobert-v1.1` (revision `551ca18`), one linear token-classification layer |
| Optimiser | AdamW, learning rate 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 |
| Schedule | linear decay to zero over the run, no warm-up |
| Batch size / max length | 8 / 512 sub-tokens; gradient clipping at norm 1.0 |
| Stopping | exactly the integer step budget; no early stopping, no checkpoint selection |
| Seeds | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 |
| Few-shot subset | first ⌊r·M⌋ sentences of a seeded permutation (nested: 2% ⊂ 5% ⊂ 10%) |
| Augmentation | K = 5 copies per mention-bearing sentence; every mention in a copy is replaced |
| Evaluation | entity-level exact match (conlleval convention), overall and for seen / unseen mentions |

**Step budgets.** Budgets are defined per seed. With *n* the few-shot size, *n*_elig the number of mention-bearing sentences and B = 8:

```
S_base = ceil(n / B) * 5                     # Base-E5
S_aug  = ceil((n + K * n_elig) / B) * 5      # Base-match, Duplicate, MR, COSINER
```

**Training-budget sweep (2% cells only).** Base, MR and COSINER are trained at 1×, 2×, 8× and 16× S_base and at S_aug. The budget of each method is selected per seed on a few-shot development set of the same size as the training subset, drawn from the validation split.

**Determinism.** All runs use deterministic PyTorch algorithms, deterministic cuDNN kernels, a fixed cuBLAS workspace and eager attention. The default memory-efficient attention kernel is not used, because its backward pass is non-deterministic on the T4. At the start of every session, a short training run is executed twice and the weights are compared bit for bit; this check is recorded in `meta.json`. Repeated runs reproduce F1 exactly on the same hardware and software.

## Installation

```bash
pip install -r requirements.txt
```

The reported runs used Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8, cuDNN 9.10.2), Transformers 5.0.0, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3 and scikit-learn 1.6.1 on an NVIDIA Tesla T4 (Kaggle Notebooks). The analysis additionally uses statsmodels and matplotlib.

## Reproducing the experiments

### 1. Run the experiments (GPU)

Edit the `USER SETTINGS` block at the top of a script:

| Setting | Meaning |
|---|---|
| `CELLS_TO_RUN` | few-shot cells for this session, e.g. `['2%']`, then `['5%']`, then `['10%']` |
| `PHASES` | `1` primary configurations, `2` budget sweep (2% only), `3` repeat runs (2% only) |
| `SEED_SHARD` | `None` for all seeds, or a list of seeds to split the work over sessions |
| `DRY_RUN` | `True` prints the run plan and the number of optimizer steps without training |
| `USE_AMP` | keep `False` for strict step parity (the reported runs used `False`) |

**Locally:**

```bash
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./outputs python steps_ncbi.py
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./outputs python steps_bc5cdr.py
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./outputs python steps_bc2gm.py
```

**On Kaggle:** add the data as notebook Input, paste the script into one cell and run it. A session stops cleanly before the 12-hour limit (`TIME_BUDGET_H`). To continue, attach the previous output as Input and run again; finished runs are skipped and outputs are merged.

Approximate cost on one T4, without AMP: about 8 GPU hours for NCBI-Disease, about 9 for BC5CDR-Chemical and about 17 for BC2GM. The exact number of runs and steps is printed by `DRY_RUN`.

### 2. Run the statistical analysis (CPU)

```bash
STEPS_ANALYSIS_ROOTS=./outputs STEPS_ANALYSIS_OUT=./analysis python analysis_steps_v2.py
```

The script finds every `steps_<corpus>/` folder below the given root, whatever the surrounding folder names are. Rows from other code versions are ignored, and duplicate copies of a run are resolved automatically. It writes `report.txt`, all tables as CSV, and Figures 6 and 7.

### 3. Draw the remaining figures (CPU)

```bash
python make_figures.py --results outputs --analysis analysis --out figures
```

## Outputs

**Experiment outputs**, in `outputs/steps_<corpus>/`:

| File | Content |
|---|---|
| `results.csv` | one row per run: configuration, budget, integer steps, effective epochs, F1 / precision / recall (overall, seen, unseen), development-set F1, training-set size, hashes of the subset and training set, session, GPU, code version |
| `counts_<cell>_s<seed>.npz` | per-test-sentence TP / FP / FN (overall, seen, unseen) for every run; used by the two-way cluster bootstrap |
| `aug_<cell>_s<seed>.json.gz` | identifiers of the few-shot subset and the development set, and the complete MR and COSINER augmented training sets |
| `aug_stats.csv` | replacement statistics: vocabulary size, candidates per mention, copies, failed attempts, change in mention length |
| `meta.json` | per session: software versions, GPU, model revision, tokenizer, data checksums, removed rows, determinism check |

Duplicate training sets are not stored, because the released code reproduces them deterministically.

**Analysis outputs**, in `analysis/`, mapped to the article:

| Article | File |
|---|---|
| Table 2, Supplementary Table S1 | `table_steps_per_seed.csv`, `table_steps_summary.csv` |
| Table 3 | `table_replacement_stats.csv` |
| Tables 4–6, Figure 4 | `table_main_results.csv` |
| Table 7, Figure 5 | `table_decomposition_cells.csv` |
| Table 8, Figure 6 | `table_sweep_curves.csv`, `table_sweep_per_budget.csv`, `table_sweep_curve_shape.csv`, `fig_step_curves.png` |
| Table 9 | `table_dev_selected_corpus.csv`, `table_dev_selected_cells.csv`, `table_dev_selection_per_seed.csv`, `table_oracle_peaks.csv` |
| Table 10, Figure 7 | `table_corpus_level.csv`, `fig_equivalence.png` |
| Table 11, Supplementary Table S2 | `table_per_cell_tests.csv` |
| Table 12, Figure 8 | `table_pr_shift_cells.csv`, `table_corpus_level.csv` |
| Figure 9 | computed by `make_figures.py` from `results.csv` |
| Sensitivity analysis (mixed model) | `table_mixed_model.csv` |
| Repeatability | `table_repeatability.csv` |

## Statistical analysis in brief

- **Per cell (descriptive).** Wilcoxon signed-rank tests on ten paired differences, and a Wilcoxon-based TOST with an equivalence margin of 1.0 F1 point. Holm correction is applied within each cell to both p-values.
- **Per corpus (inference).** Within a corpus, the cells are dependent: subsets are nested and the test set is shared. Each corpus is therefore analysed with a two-way cluster bootstrap (B = 2000) that resamples seeds, each carrying its three nested cells, and test sentences, recomputing F1 from per-sentence counts. Difference tests are Holm-adjusted within each corpus, and equivalence holds when the 90% interval lies within ±1.0 point.
- **Sensitivity.** Margins of 0.5 and 1.5 points, and a linear mixed model with a random seed intercept.
- **Nine-cell averages.** Reported as descriptive only.

## Citation

If you use this code or data, please cite the article (citation details will be updated after publication):

```bibtex
@article{siswanto_more_steps,
  title   = {More Steps, Not More Data: Re-Examining Data Augmentation for Few-Shot Biomedical Named Entity Recognition},
  author  = {Siswanto, Agus and Tutuko, Bambang and Firdaus and Jasmir},
  journal = {Computers, Materials \& Continua},
  note    = {Under review}
}
```

## License

The code is released under the [LICENSE](LICENSE) of this repository. The benchmark corpora remain under the terms of their original distributions.

## Contact

Corresponding author: Bambang Tutuko (bambang_tutuko@unsri.ac.id), Faculty of Computer Science, Universitas Sriwijaya, Indonesia.
