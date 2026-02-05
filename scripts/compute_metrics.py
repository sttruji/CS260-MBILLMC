"""
Stage 2: Volume & Code Metrics
Reads data/processed/pr_details.csv and writes data/processed/processed_data.csv
with added columns: commit_count, nloc_total, avg_nloc_per_commit, commit_stats_json
Supports incremental processing & resume.
"""
import os
import pandas as pd
import json
import argparse
from extractors.metrics import MetricsExtractor
from extractors.github_utils import GitHubUtils

OUTPUT = 'data/processed/processed_data.csv'

def load_processed_prs(output=OUTPUT):
    if os.path.exists(output):
        df = pd.read_csv(output, usecols=['repo_name', 'pr_number'])
        return set((r, int(n)) for r, n in zip(df['repo_name'], df['pr_number']))
    return set()

def append_rows(rows, output=OUTPUT):
    os.makedirs('data/processed', exist_ok=True)
    df_new = pd.DataFrame(rows)
    if os.path.exists(output):
        df_existing = pd.read_csv(output)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output, index=False)
    else:
        df_new.to_csv(output, index=False)

def main(input_csv='data/processed/pr_details.csv', max_commits=None):
    if not os.path.exists(input_csv):
        print(f"{input_csv} missing. Run Stage 1 first.")
        return

    pr_df = pd.read_csv(input_csv)
    processed = load_processed_prs()
    github_utils = GitHubUtils()
    extractor = MetricsExtractor()

    rows_to_append = []

    for idx, row in pr_df.iterrows():
        key = (row['repo_name'], int(row['pr_number']))
        if key in processed:
            continue  # skip already-processed PR

        repo = github_utils.get_repository(row['repo_name'])
        if not repo:
            print(f"  ✗ Unable to fetch repo {row['repo_name']} - skipping PR {row['pr_number']}")
            continue

        try:
            pr_obj = repo.get_pull(int(row['pr_number']))
        except Exception as e:
            print(f"  ✗ Failed to fetch PR {row['pr_number']} in {row['repo_name']}: {e}")
            continue

        metrics = extractor.compute_pr_metrics(repo, pr_obj, max_commits=max_commits)

        # Build combined row
        combined = row.to_dict()
        combined.update({
            'commit_count': metrics['commit_count'],
            'nloc_total': metrics['nloc_total'],
            'avg_nloc_per_commit': metrics['avg_nloc_per_commit'],
            'commit_stats_json': json.dumps(metrics['commit_stats'])
        })
        rows_to_append.append(combined)

        # Save incrementally every 10 PRs to reduce data loss and memory
        if len(rows_to_append) >= 10:
            append_rows(rows_to_append)
            processed.update((r['repo_name'], int(r['pr_number'])) for r in rows_to_append)
            rows_to_append = []

    # Final flush
    if rows_to_append:
        append_rows(rows_to_append)
    print("Stage 2 complete. Output:", OUTPUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-commits', type=int, default=None,
                        help='Optional limit of commits per PR to inspect (speeds up run)')
    args = parser.parse_args()
    main(max_commits=args.max_commits)