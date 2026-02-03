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
    Mining script has been running for ~20 hours. We have identified ~10 candidate repos that match our specific criteria:
    Language: Python (critical for our downstream Pylint/ODC analysis).
    Activity: >200 Stars, Pushed since Jan 1, 2024.
    Data Density: >50 Total PRs, with at least 10+ AI-tagged PRs (to ensure we have a control group).
*Note*: Treat these as candidates. They fit the search parameters, but we still need to verify they contain usable data for our deep dive.

Immediate Plan:
    I am updating the mining script to cache results (skip already-checked repos) to save time. This should be live by tomorrow.

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

