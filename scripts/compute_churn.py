import os, json
import pandas as pd
import argparse
from extractors.churn_analyzer import ChurnAnalyzer
from extractors.github_utils import GitHubUtils

OUTPUT = 'data/processed/rework_metrics.csv'

def load_processed(output=OUTPUT):
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

def main(input_csv='data/processed/processed_data.csv', window_days=90, max_files=None):
    if not os.path.exists(input_csv):
        print("Run Stage 2 first.")
        return

    df = pd.read_csv(input_csv)
    processed = load_processed()
    gha = GitHubUtils()
    analyzer = ChurnAnalyzer()

    rows = []
    for idx, row in df.iterrows():
        key = (row['repo_name'], int(row['pr_number']))
        if key in processed:
            continue

        repo = gha.get_repository(row['repo_name'])
        if not repo:
            print(f" ✗ Cannot fetch repo {row['repo_name']}")
            continue

        try:
            pr = repo.get_pull(int(row['pr_number']))
            churn = analyzer.compute_pr_rework(repo, pr, window_days=window_days, max_files=max_files)
            out = {
                'repo_name': row['repo_name'],
                'pr_number': int(row['pr_number']),
                'merged_at': churn['merged_at'] if churn else None,
                'commit_count': row.get('commit_count', row.get('commits', None)),
                'churn_lines_90d': churn['churn_lines_90d'] if churn else 0,
                'rework_events': churn['rework_events'] if churn else 0,
                'files_reworked_count': churn['files_reworked_count'] if churn else 0,
                'first_rework_date': churn['first_rework_date'] if churn else None,
                'last_rework_date': churn['last_rework_date'] if churn else None,
                'per_file_rework_json': json.dumps(churn['per_file_rework'] if churn else [])
            }
            rows.append(out)
        except Exception as e:
            print(f" ✗ Failed PR {row['pr_number']} in {row['repo_name']}: {e}")
            continue

        if len(rows) >= 10:
            append_rows(rows)
            rows = []

    if rows:
        append_rows(rows)

    print("Stage 3 complete. Output:", OUTPUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-days', type=int, default=90)
    parser.add_argument('--max-files', type=int, default=None, help='Limit number of files per PR to inspect')
    args = parser.parse_args()
    main(window_days=args.window_days, max_files=args.max_files)