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
                       = [F1(MR) − F1(Duplicate)] + [F1(Duplicate) − F1(Base-match)]
```

Base-E5 is the reference protocol: five epochs on the original data. The content component is an operational difference: at equal steps, the augmented and Base-match training sets differ in composition and sample exposure as well as in content. Duplicate shares the composition and exposure of the augmented sets, so the second line separates the replaced mentions from these differences (article, Sections 3.2, 3.5 and 5.2).

## Repository structure

```
Steps_FewShot_BioNER/
├── README.md
├── requirements.txt
├── steps_ncbi.py            # experiments, NCBI-Disease
├── steps_bc5cdr.py          # experiments, BC5CDR-Chemical
├── steps_bc2gm.py           # experiments, BC2GM
├── analysis_steps.py        # statistical analysis: Tables 2–3 and 7–12, Figures 6–7, report.txt
├── make_figures.py          # Figures 1–5, 8 and 9
├── check_truncation.py      # audit of the 512-sub-token input limit (Section 4.6)
├── Dataset/                 # exact benchmark files used (see "Data")
│   ├── ncbi_disease/        ncbi_disease_train.csv  ncbi_disease_validation.csv  ncbi_disease_test.csv
│   ├── bc5cdr/              bc5cdr_train.csv  bc5cdr_validation.csv  bc5cdr_test.csv
│   └── bc2gm_corpus/        bc2gm_corpus_train.csv  bc2gm_corpus_validation.csv  bc2gm_corpus_test.csv
├── result_ncbi_2%/          # raw outputs of one Kaggle session (one few-shot cell); see "Outputs"
├── result_ncbi_5%/
├── result_ncbi_10%/
├── result_bc5cdr_2%/  result_bc5cdr_5%/  result_bc5cdr_10%/
├── result_bc2gm_2%/   result_bc2gm_5%/   result_bc2gm_10%/
└── analysis/                # output of analysis_steps.py and check_truncation.py
```

The experiments were run on Kaggle, one session per corpus and few-shot cell. In each session, the scripts read the benchmark files as `<corpus>/train.csv`, `validation.csv` and `test.csv` and wrote their outputs to `steps_<corpus>/` as `results.csv`, `meta.json`, `aug_stats.csv`, `counts_*.npz` and `aug_*.json.gz`. For this repository, the benchmark files and the per-session result files were renamed so that the nine sessions could be stored side by side (for example, `results.csv` of the NCBI-Disease 2% session is `result_ncbi_2%/resultssteps_ncbi_2%.csv`). The scripts were not changed and still expect the original names, so the released files are first copied once to those names (next section).

## Quick start: reproduce all tables and figures (CPU, about two minutes)

The released outputs contain every run, so the analysis can be repeated without a GPU. Run the commands from the repository root in a Bash shell (Linux, macOS, Kaggle or Colab; on Windows, Git Bash or WSL).

**1. Install the CPU packages:**

```bash
git clone https://github.com/wantodbz/Steps_FewShot_BioNER.git
cd Steps_FewShot_BioNER
pip install numpy==2.0.2 pandas==2.3.3 scipy==1.16.3 statsmodels matplotlib "transformers==5.0.0"
```

**2. Prepare the layout that the scripts read (once).** This copies the released files to the names that the scripts used on Kaggle, in two new folders, `data/` and `outputs/`; the files in `Dataset/` and `result_*/` are not modified.

```bash
mkdir -p data/ncbi-disease data/bc5cdr data/bc2gm-corpus
for s in train validation test; do
  cp "Dataset/ncbi_disease/ncbi_disease_$s.csv" "data/ncbi-disease/$s.csv"
  cp "Dataset/bc5cdr/bc5cdr_$s.csv"             "data/bc5cdr/$s.csv"
  cp "Dataset/bc2gm_corpus/bc2gm_corpus_$s.csv" "data/bc2gm-corpus/$s.csv"
done
for c in ncbi bc5cdr bc2gm; do
  for cell in 2% 5% 10%; do
    src="result_${c}_${cell}"; dst="outputs/steps_${c}/${cell}"
    mkdir -p "$dst"
    cp "$src"/*esults*steps*.csv "$dst/results.csv"
    cp "$src"/*eta*steps*.json   "$dst/meta.json"
    cp "$src"/aug_stats.csv "$src"/counts_*.npz "$src"/aug_*.json.gz "$dst/"
  done
done
```

**3. Run the analysis:**

```bash
# statistics, tables (CSV), Figures 6 and 7, report.txt  ->  analysis/
STEPS_ANALYSIS_ROOTS=./outputs STEPS_ANALYSIS_OUT=./analysis python analysis_steps.py

# Figures 1-5, 8 and 9  ->  figures/
python make_figures.py --results outputs --analysis analysis --out figures

# truncation audit (Section 4.6)  ->  analysis/table_truncation*.csv
python check_truncation.py --data ./data --outputs ./outputs --out ./analysis
```

The first command rewrites the CSV tables and `report.txt` in `analysis/` with identical content (checked from a fresh clone with Python 3.12, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3, statsmodels 0.15.0 and Matplotlib 3.11.2); the PNG/PDF files of Figures 6 and 7 can differ in their binary encoding, for example in font rendering, but not in content. The second command creates `figures/`. The third command downloads the BioBERT tokenizer from the Hugging Face Hub; offline, pass the BioBERT v1.1 vocabulary instead with `--vocab /path/to/vocab.txt`.

## Data

The three corpora were obtained from the Hugging Face Hub with their original training, validation and test splits. The exact files used are included in `Dataset/`, and their MD5 checksums were recorded automatically in every experimental session (`meta*.json` in each `result_*` folder).

| Corpus | Entity type | Hugging Face dataset | Train / Validation / Test (rows) |
|---|---|---|---|
| NCBI-Disease | Disease | `ncbi_disease` | 5,433 / 924 / 941 |
| BC5CDR | Chemical (disease labels mapped to O) | `tner/bc5cdr` | 5,228 / 5,330 / 5,865 |
| BC2GM | Gene | `bc2gm_corpus` | 12,500 / 2,500 / 5,000 |

| File in this repository | Name read by the scripts | MD5 |
|---|---|---|
| `Dataset/ncbi_disease/ncbi_disease_train.csv` | `ncbi-disease/train.csv` | `9478cf21fb677bf49ee7654dd38643a2` |
| `Dataset/ncbi_disease/ncbi_disease_validation.csv` | `ncbi-disease/validation.csv` | `38bb954b3433892bb6ed04c08c48b67c` |
| `Dataset/ncbi_disease/ncbi_disease_test.csv` | `ncbi-disease/test.csv` | `d317d4fca3309df15683c8ac7657a893` |
| `Dataset/bc5cdr/bc5cdr_train.csv` | `bc5cdr/train.csv` | `25842c1c3731c6bc5c384b057580b698` |
| `Dataset/bc5cdr/bc5cdr_validation.csv` | `bc5cdr/validation.csv` | `001214ffbc07427ef3ce4745f7b20952` |
| `Dataset/bc5cdr/bc5cdr_test.csv` | `bc5cdr/test.csv` | `bdee78255cc327506e26593cfaa34590` |
| `Dataset/bc2gm_corpus/bc2gm_corpus_train.csv` | `bc2gm-corpus/train.csv` | `342d73d895cc7b2fe221d93fc90e9b17` |
| `Dataset/bc2gm_corpus/bc2gm_corpus_validation.csv` | `bc2gm-corpus/validation.csv` | `edafb85f5fe3ee4080363eb171236579` |
| `Dataset/bc2gm_corpus/bc2gm_corpus_test.csv` | `bc2gm-corpus/test.csv` | `7caaa93fc44b8845a9c61b229cf93b95` |

Each NCBI-Disease file ends with one empty row, which the parser removes. No other rows are removed. Please cite the original corpora when using them:

- NCBI-Disease: Doğan et al., *J Biomed Inform* 2014.
- BC5CDR: Li et al., *Database* 2016.
- BC2GM: Smith et al., *Genome Biol* 2008.

## Experimental protocol

| Item | Setting |
|---|---|
| Encoder | `dmis-lab/biobert-v1.1` (revision `551ca18`), one linear token-classification layer |
| Tokenizer | cased BioBERT WordPiece (vocabulary 28,996); at most 512 sub-tokens including [CLS] and [SEP] |
| Optimiser | AdamW, learning rate 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 |
| Schedule | linear decay to zero over the run, no warm-up |
| Batch size | 8; gradient clipping at norm 1.0 |
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

**Input length.** Every sample, original or augmented, is encoded separately with `truncation=True, max_length=512`, and labels are attached to the retained sub-tokens after truncation. `check_truncation.py` rebuilds every training set, checks it against the training-set hash recorded for each of the 798 runs, and counts the samples above the limit: none of the original, Duplicate, MR or COSINER training samples, development sentences or test sentences exceeds it (longest sample: 477 of 512 sub-tokens).

**Determinism.** All runs use deterministic PyTorch algorithms, deterministic cuDNN kernels, a fixed cuBLAS workspace and eager attention. The default memory-efficient attention kernel is not used, because its backward pass is non-deterministic on the T4. At the start of every session, a short training run is executed twice and the weights are compared bit for bit; this check is recorded in the session's meta file. Repeated runs reproduce F1 exactly on the same hardware and software.

## Installation

```bash
pip install -r requirements.txt
```

The reported runs used Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8, cuDNN 9.10.2), Transformers 5.0.0, NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.3 and scikit-learn 1.6.1 on an NVIDIA Tesla T4 (Kaggle Notebooks). The analysis additionally uses statsmodels and Matplotlib; for the analysis only, the CPU packages of the quick start are sufficient.

## Re-running the experiments (GPU)

Edit the `USER SETTINGS` block at the top of a script:

| Setting | Meaning |
|---|---|
| `CELLS_TO_RUN` | few-shot cells for this session, e.g. `['2%']`, then `['5%']`, then `['10%']` |
| `PHASES` | `1` primary configurations, `2` budget sweep (2% only), `3` repeat runs (2% only) |
| `SEED_SHARD` | `None` for all seeds, or a list of seeds to split the work over sessions |
| `DRY_RUN` | set the environment variable `STEPS_DRY_RUN=1` (or `DRY_RUN = True`) to print the run plan and the number of optimizer steps without training |
| `USE_AMP` | keep `False` for strict step parity (the reported runs used `False`) |

**Locally**, after step 2 of the quick start, write to a new output folder:

```bash
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./rerun python steps_ncbi.py
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./rerun python steps_bc5cdr.py
STEPS_DATA_ROOT=./data STEPS_OUT_DIR=./rerun python steps_bc2gm.py
```

The scripts resume from whatever their output folder already contains, so a new folder such as `./rerun` repeats the experiments from scratch. The analysis of the new runs is obtained with `STEPS_ANALYSIS_ROOTS=./rerun STEPS_ANALYSIS_OUT=./analysis_rerun python analysis_steps.py`.

**On Kaggle:** upload the prepared `data/` folder as a dataset, keeping its sub-folders `ncbi-disease/`, `bc5cdr/` and `bc2gm-corpus/` with `train.csv`, `validation.csv` and `test.csv`. Add it as notebook Input, paste the script into one cell and run it; the script finds the files below `/kaggle/input` and writes to `/kaggle/working/steps_<corpus>/`. A session stops cleanly before the 12-hour limit (`TIME_BUDGET_H`). To continue, attach the previous output as Input and run again; finished runs are skipped and outputs are merged.

Approximate cost on one T4, without AMP: about 8 GPU hours for NCBI-Disease, about 9 for BC5CDR-Chemical and about 17 for BC2GM. The exact number of runs and steps is printed by the dry run.

## Outputs

**Experiment outputs.** Each folder `result_<corpus>_<cell>/` holds the outputs of the Kaggle session that ran that few-shot cell:

| File in `result_<corpus>_<cell>/` | Name written by the scripts | Content |
|---|---|---|
| `resultssteps_<corpus>_<cell>.csv` (`results_steps_bc2gm_2%.csv` in `result_bc2gm_2%/`) | `results.csv` | one row per run: configuration, budget, integer steps, effective epochs, F1 / precision / recall (overall, seen, unseen), development-set F1, training-set size, hashes of the subset and training set, number of test words without a retained sub-token, session, GPU, code version |
| `metasteps_<corpus>_<cell>.json` (`meta_steps_bc2gm_2%.json` in `result_bc2gm_2%/`) | `meta.json` | software versions, GPU, model revision, tokenizer, data paths and checksums, removed rows, determinism check |
| `counts_<cell>_s<seed>.npz` | same | per-test-sentence TP / FP / FN (overall, seen, unseen) for every run; used by the two-way cluster bootstrap |
| `aug_<cell>_s<seed>.json.gz` | same | identifiers of the few-shot subset and the development set, and the complete MR and COSINER augmented training sets |
| `aug_stats.csv` | same | replacement statistics: vocabulary size, candidates per mention, copies, failed attempts, change in mention length |

Duplicate training sets are not stored, because the released code reproduces them deterministically; `check_truncation.py` verifies this against the recorded training-set hashes.

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
| Figure 9 | computed by `make_figures.py` from the results files |
| Section 4.6, input length and truncation | `table_truncation.csv`, `table_truncation_summary.csv` (from `check_truncation.py`) |
| Sensitivity analysis (mixed model) | `table_mixed_model.csv` |
| Repeatability | `table_repeatability.csv` |

## Statistical analysis in brief

- **Per cell (descriptive).** Wilcoxon signed-rank tests on ten paired differences, and a Wilcoxon-based TOST with an equivalence margin of 1.0 F1 point. Holm correction is applied within each cell to both p-values.
- **Per corpus (inference).** Within a corpus, the cells are dependent: subsets are nested and the test set is shared. Each corpus is therefore analysed with a two-way cluster bootstrap (B = 2000) that resamples seeds, each carrying its three nested cells, and test sentences, recomputing F1 from per-sentence counts. Difference tests are Holm-adjusted within each corpus, and equivalence holds when the 90% interval lies within ±1.0 point.
- **Sensitivity.** Margins of 0.5 and 1.5 points, and a linear mixed model with a random seed intercept.
- **Nine-cell averages.** Reported as descriptive only.

## Changes in this version (second revision)

- The README now describes the repository as uploaded (`Dataset/`, `result_<corpus>_<cell>/`), refers to `analysis_steps.py` (previously `analysis_steps_v2.py`), and adds the one-time preparation step that copies the released files to the names the scripts read.
- Added `check_truncation.py` and its output tables (`analysis/table_truncation*.csv`).
- No script, data or result file was changed.
