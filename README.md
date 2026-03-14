# CS260-MBILLMC

## Quick Reproduction

Use this path if you only need to reproduce the final RQ1 v2 and RQ2 v2 analysis notebooks for grading or review.

The repository already includes the prebuilt result files:

- `results/rq1_main_frame.parquet`
- `results/rq1_main_frame_v2.parquet`
- `results/rq1_szz_pairs.parquet`

Because those files are already checked in, you do not need to set up `.env`, call the GitHub API, or rebuild the main frames just to rerun the notebooks.

### 1. Clone the repository

```bash
git clone <repo-url>
cd CS260-MBILLMC
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

`requirements.txt` covers the core pipeline dependencies. The notebooks also need the analysis stack.

```bash
pip install -r requirements.txt
pip install matplotlib seaborn scipy statsmodels jupyterlab nbconvert ipykernel
```

### 4. Run the v2 notebooks

Interactive:

```bash
jupyter lab notebooks/rq1_v2_analysis.ipynb notebooks/rq2_v2_analysis.ipynb
```

Headless execution:

```bash
jupyter nbconvert --to notebook --execute notebooks/rq1_v2_analysis.ipynb --output rq1_v2_analysis.executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/rq2_v2_analysis.ipynb --output rq2_v2_analysis.executed.ipynb
```

Notes:

- `notebooks/rq1_v2_analysis.ipynb` reads `../results/rq1_main_frame_v2.parquet`, `../results/rq1_szz_pairs.parquet`, and `../results/rq1_main_frame.parquet`.
- `notebooks/rq2_v2_analysis.ipynb` reads `../results/rq1_main_frame_v2.parquet`.
- If you are only reproducing the submitted results, you can stop here.

## Full Recreation Guide

Use this path if you want to rebuild the data products from the raw upstream datasets and regenerate the v2 SZZ-backed outputs yourself.

### Prerequisites

- Python 3.9+ recommended
- Git
- Docker
- Java runtime available on `PATH`
- Internet access for:
  - Hugging Face parquet datasets
  - GitHub API requests
  - cloning repositories for SZZ
- A GitHub personal access token for the ingest pipeline

### 1. Clone the repository

```bash
git clone <repo-url>
cd CS260-MBILLMC
```

### 2. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib seaborn scipy statsmodels jupyterlab nbconvert ipykernel
```

### 3. Configure `.env`

Copy the template and set your GitHub token:

```bash
cp env.example .env
```

`env.example` currently expects:

```bash
GITHUB_TOKEN=YOUR_TOKEN
```

The ingest script loads `.env` automatically. Without a token, GitHub API calls will be heavily rate-limited.

### 4. Build the base RQ1 main frame

This stage creates `results/rq1_main_frame.parquet` and `results/rq1_main_frame.csv`.

Full run:

```bash
python scripts/rq1_ingest.py
```

Useful variants:

```bash
python scripts/rq1_ingest.py --resume
python scripts/rq1_ingest.py --target_count 100
python scripts/rq1_ingest.py --resume --target_count 100
python scripts/rq1_ingest.py --resume --skip-age
python scripts/rq1_ingest.py --recompute-defects
python scripts/rq1_ingest.py --recompute-loc
```

What these do:

- `--resume`: continue from `data/processed/rq1_checkpoint.parquet`
- `--target_count N`: limit the number of repositories processed
- `--skip-age`: skip the repository age step
- `--recompute-defects`: recompute only the original defect columns on an existing checkpoint
- `--recompute-loc`: recompute only `pr_size_loc` and dependent `fix_size` columns on an existing checkpoint

Key outputs and caches:

- `results/rq1_main_frame.parquet`
- `results/rq1_main_frame.csv`
- `data/processed/rq1_checkpoint.parquet`

### 5. Build the v2 SZZ-backed main frame

This stage consumes `results/rq1_main_frame.parquet`, generates SZZ issue lists and bug-introducing/fixing pairs, and writes the v2 dataset.

Full run with conservative checkpointing:

```bash
python scripts/rq1_v2.py --resume --checkpoint-every 1 --szz-timeout 60
```

Useful variants:

```bash
python scripts/rq1_v2.py --resume
python scripts/rq1_v2.py --repo "microsoft/TypeScript"
python scripts/rq1_v2.py --limit-repos 25
python scripts/rq1_v2.py --resume --limit-repos 25 --checkpoint-every 1
python scripts/rq1_v2.py --resume --refresh-repos
python scripts/rq1_v2.py --resume --refresh-szz
python scripts/rq1_v2.py --resume --cleanup-repos
python scripts/rq1_v2.py --resume --checkpoint-every 1 --szz-timeout 120
```

What these do:

- `--resume`: reuse `data/processed/rq1_v2_checkpoint.parquet` and any cached SZZ outputs
- `--repo "owner/repo"`: process a single repository cohort
- `--limit-repos N`: process only the first `N` repositories
- `--refresh-repos`: refetch cached Git repositories before SZZ analysis
- `--refresh-szz`: rerun SZZ even if cached results already exist
- `--cleanup-repos`: delete cloned repositories after processing to save disk
- `--checkpoint-every N`: save the v2 checkpoint every `N` repositories
- `--szz-timeout M`: stop SZZ after `M` minutes per repository and keep partial results when possible

Important implementation details:

- `scripts/rq1_v2.py` builds SZZUnleashed with Docker if a jar is not already available.
- It then runs the jar locally with `java -jar ...`, so Java must also be installed.
- Repository clones and SZZ artifacts are cached under `data/cache/`.

Key outputs and caches:

- `results/rq1_main_frame_v2.parquet`
- `results/rq1_main_frame_v2.csv`
- `results/rq1_szz_pairs.parquet`
- `data/processed/rq1_v2_checkpoint.parquet`
- `data/cache/repos/`
- `data/cache/szz_issue_lists/`
- `data/cache/szz_results/`

Optional: if you already have a built SZZ jar, you can point the pipeline at it:

```bash
export RQ1_V2_SZZ_JAR=/absolute/path/to/szz_find_bug_introducers-<version>.jar
python scripts/rq1_v2.py --resume
```

### 6. Execute the analysis notebooks

After rebuilding the datasets, rerun the notebooks so figures and model outputs are regenerated from the new parquet files.

Interactive:

```bash
jupyter lab notebooks/rq1_v2_analysis.ipynb notebooks/rq2_v2_analysis.ipynb
```

Headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/rq1_v2_analysis.ipynb --output rq1_v2_analysis.executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/rq2_v2_analysis.ipynb --output rq2_v2_analysis.executed.ipynb
```

## Recommended Order

If you are rebuilding everything from scratch, run the project in this order:

1. Set up the virtual environment and install dependencies.
2. Create `.env` with `GITHUB_TOKEN`.
3. Run `python scripts/rq1_ingest.py --resume`.
4. Run `python scripts/rq1_v2.py --resume --checkpoint-every 1 --szz-timeout 60`.
5. Run `notebooks/rq1_v2_analysis.ipynb`.
6. Run `notebooks/rq2_v2_analysis.ipynb`.

## Repository Notes

- The v1 notebooks are still present:
  - `notebooks/rq1_analysis.ipynb`
  - `notebooks/rq2_analysis.ipynb`
- The submitted v2 analysis notebooks are:
  - `notebooks/rq1_v2_analysis.ipynb`
  - `notebooks/rq2_v2_analysis.ipynb`
- For a quick grading pass, the v2 notebooks are the ones you want.
