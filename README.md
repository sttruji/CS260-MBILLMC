# Setup 

**Environment** 

python -m venv .venv 

source .venv/bin/activate 

pip install -r requirements.txt 

cp env.example .env

Github > Settings > Developer Settings > Personal access token > Tokens (Classic)
Generate new token > Scope (public_repo / read:user)

Place key in .env 

**Scripts** 

python scripts/collect_github.py 
    Collects repositories from github with the following critera 
        200+ Stars on github
        Python based Repository (needed downstream)
        50+ PRs 
        Pushed since Jan 1, 2024
        10+ AI tagged PRs (Keyword search may not be accurate)
        or has AI contribution roles in readme / contributions files (may not be accurate)

**Next Steps**

Project Update & Next Steps

Status:
    Mining script has been running for ~20 hours. We have identified 9 candidate repos that match our specific criteria:
    Language: Python (critical for our downstream Pylint/ODC analysis).
    Activity: >200 Stars, Pushed since Jan 1, 2024.
    Data Density: >50 Total PRs, with at least 10+ AI-tagged PRs (to ensure we have a control group).
*Note*: Treat these as candidates. They fit the search parameters, but we still need to verify they contain usable data for our deep dive.

Immediate Update:
    The mining script cahces results (skip already-checked repos) to save time. This should be live.

Action Items (Who can take these?):
1. Data Processing Script
    We need a script to ingest these candidate repos and output a processed_data.csv with the following columns:
    Classification: Tag AI vs. Human (based on "AI-Generated" labels or body text).
    Timestamps: Capture dates to infer model usage (Copilot vs. Cursor).
    Volume: Calculate NLOC per commit/PR (so we can compare similar-sized work).
    Rework Metrics:
    Churn: Lines re-modified within 90 days.
    Retries: Count of commits/iterations before PR acceptance.




2. Comparative Analysis Script
    Once we have the CSV, we need a script to run the stats:
    Compare Productivity (Speed, Volume) vs. Maintenance (Rework, Churn).
    Goal: See if the "productivity boost" correlates with higher maintenance debt.
    Let’s aim to have preliminary numbers for the Methodology section of the report!

---

## Data Processing Pipeline (Implemented) ✅

- **Stage 1 — PR Ingestion & Classification** 🔧
  - Files: `scripts/extractors/pr_classifier.py`, `scripts/extractors/github_utils.py`, `scripts/process_data.py`
  - Output: `data/processed/pr_details.csv`
  - Description: Fetches merged PRs and classifies PRs as AI vs Human via labels, keywords, and author patterns. Supports incremental saves/resume.

- **Stage 2 — Volume & Code Metrics** 🧮
  - Files: `scripts/extractors/metrics.py`, `scripts/compute_metrics.py`
  - Output: `data/processed/processed_data.csv`
  - Description: Computes commit-level NLOC, per-PR total and average NLOC, and commit counts. Supports `--max-commits` flag to limit API calls.

- **Stage 3 — Rework & Churn Metrics** 🔍
  - Files: `scripts/extractors/churn_analyzer.py`, `scripts/compute_churn.py`
  - Output: `data/processed/rework_metrics.csv` and cached JSONs in `data/cache/rework/`
  - Description: Checks whether files modified by a PR were re-modified within 90 days and computes churn lines and rework events. Supports `--window-days` and `--max-files` flags.

---

## How to run (Step-by-step) ▶️


1. Stage 1 — Ingest & classify PRs
   - `python scripts/process_data.py`
   - Output: `data/processed/pr_details.csv`

2. Stage 2 — Compute commit-level metrics (NLOC)
   - Test run: `python scripts/compute_metrics.py --max-commits 50`
   - Output: `data/processed/processed_data.csv`

3. Stage 3 — Compute churn / rework metrics
   - Test run: `python scripts/compute_churn.py --window-days 90 --max-files 5`
   - Output: `data/processed/rework_metrics.csv` and `data/cache/rework/`

---

### Notes & Tips ⚠️
- Add `pandas` and `PyGithub` to `requirements.txt`. Consider `GitPython` for precise git-based churn later.
- Scripts include retry/backoff and incremental saving; re-run picks up where it left off.
- Use `--max-commits` and `--max-files` to limit API usage. Run long jobs in `tmux` or `nohup`.
