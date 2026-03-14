"""Diagnostic script: trace where SZZ signal is lost in the rq1_v2 pipeline."""
import json, os, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_v2 import (
    SZZ_RESULT_DIR, SZZ_ISSUE_LIST_DIR, safe_repo_slug,
    load_szz_pairs, BASE_DIR
)
import pandas as pd

# ── 1. How many repos have SZZ results at all? ──
print("=" * 70)
print("STAGE 1: SZZ EXECUTION COVERAGE")
print("=" * 70)

result_dirs = sorted(SZZ_RESULT_DIR.iterdir()) if SZZ_RESULT_DIR.exists() else []
print(f"\nTotal repo result directories: {len(result_dirs)}")

has_main_result = 0
has_partitioned = 0
has_issues_only = 0
has_empty_result = 0
total_raw_pairs = 0
repos_with_pairs = 0
pair_counts = {}

for d in result_dirs:
    repo_slug = d.name
    main_file = d / "results" / "fix_and_introducers_pairs.json"
    issues_dir = d / "issues"
    results_dir = d / "results"
    
    if main_file.exists():
        has_main_result += 1
        pairs = load_szz_pairs(main_file)
        pair_counts[repo_slug] = len(pairs)
        total_raw_pairs += len(pairs)
        if pairs:
            repos_with_pairs += 1
        else:
            has_empty_result += 1
    elif results_dir.exists() and list(results_dir.glob("result*/")):
        has_partitioned += 1
    elif issues_dir.exists():
        has_issues_only += 1

print(f"  Repos with merged result file: {has_main_result}")
print(f"  Repos with partitioned (unmerged): {has_partitioned}")
print(f"  Repos with only issues/ (no results): {has_issues_only}")
print(f"  Repos with EMPTY result (0 pairs): {has_empty_result}")
print(f"  Repos with ≥1 raw SZZ pair: {repos_with_pairs}")
print(f"  Total raw SZZ pairs across all repos: {total_raw_pairs}")

print(f"\nTop 20 repos by raw pair count:")
for slug, count in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {slug}: {count} pairs")

# ── 2. How many repos have issue lists? (were eligible for SZZ) ──
print("\n" + "=" * 70)
print("STAGE 2: SZZ ELIGIBILITY (issue lists generated)")
print("=" * 70)

issue_lists = sorted(SZZ_ISSUE_LIST_DIR.glob("*_issue_list.json")) if SZZ_ISSUE_LIST_DIR.exists() else []
print(f"\nTotal issue list files: {len(issue_lists)}")

issue_list_sizes = {}
for f in issue_lists:
    with open(f) as h:
        data = json.load(h)
    slug = f.stem.replace("_issue_list", "")
    issue_list_sizes[slug] = len(data)

non_empty = sum(1 for v in issue_list_sizes.values() if v > 0)
print(f"  Non-empty issue lists: {non_empty}")
print(f"  Empty issue lists: {len(issue_lists) - non_empty}")
print(f"  Total fix commit entries across all lists: {sum(issue_list_sizes.values())}")

# ── 3. Load the main frame & pair trace to check matching ──
print("\n" + "=" * 70)
print("STAGE 3: PAIR TRACE MATCHING FUNNEL")
print("=" * 70)

df = pd.read_parquet(str(BASE_DIR / "results" / "rq1_main_frame_v2.parquet"))
pairs = pd.read_parquet(str(BASE_DIR / "results" / "rq1_szz_pairs.parquet"))

print(f"\nFinal pair trace rows: {len(pairs)}")
print(f"PRs with has_defect_90d=1: {(df['has_defect_90d'] == 1).sum()}")

# For repos that had raw SZZ pairs, compare raw vs matched
print(f"\nPer-repo raw-vs-matched comparison (repos with raw pairs):")
for slug, raw_count in sorted(pair_counts.items(), key=lambda x: -x[1]):
    if raw_count == 0:
        continue
    # Convert slug back to repo name (approximate)
    matched = pairs[pairs["repo_full_name"].apply(lambda x: safe_repo_slug(x)) == slug]
    print(f"  {slug}: {raw_count} raw SZZ pairs → {len(matched)} matched to PRs")

# ── 4. Deep dive: why do pairs get dropped? ──
print("\n" + "=" * 70)
print("STAGE 4: DEEP DIVE INTO A HIGH-PAIR REPO")
print("=" * 70)

# Pick the repo with the most raw pairs that we can inspect
if pair_counts:
    top_slug = max(pair_counts, key=pair_counts.get)
    top_repo = None
    for r in df["repo_full_name"].unique():
        if safe_repo_slug(r) == top_slug:
            top_repo = r
            break
    
    if top_repo:
        print(f"\nInspecting: {top_repo} ({pair_counts[top_slug]} raw pairs)")
        
        # Load raw pairs
        raw_file = SZZ_RESULT_DIR / top_slug / "results" / "fix_and_introducers_pairs.json"
        raw_pairs = load_szz_pairs(raw_file)
        
        # Get the repo's PRs
        repo_frame = df[df["repo_full_name"] == top_repo]
        print(f"  PRs in dataset for this repo: {len(repo_frame)}")
        print(f"  Merged PRs: {repo_frame['merged_at'].notna().sum()}")
        print(f"  Candidate fix PRs: {(repo_frame['szz_candidate_fix'] == 1).sum()}")
        
        # Get PR commit SHAs
        pr_commits = pd.read_parquet(str(BASE_DIR / "results" / "rq1_szz_pairs.parquet"))
        
        # Load the pr_commits data for this repo
        pr_ids = repo_frame["id"].astype(int).tolist()
        
        # Get all unique SHAs from raw pairs
        introducing_shas = {p[0] for p in raw_pairs}
        fixing_shas = {p[1] for p in raw_pairs}
        all_pair_shas = introducing_shas | fixing_shas
        
        print(f"\n  Unique bug-introducing SHAs from SZZ: {len(introducing_shas)}")
        print(f"  Unique fixing SHAs from SZZ: {len(fixing_shas)}")
        
        # Check: how many of these SHAs are in our PR commit dataset?
        # Load pr_commits from HF cache or parquet
        try:
            pr_commits_all = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commits.parquet",
                                              columns=["pr_id", "sha"],
                                              filters=[("pr_id", "in", pr_ids)])
            pr_shas = set(pr_commits_all["sha"].dropna())
            print(f"  PR commit SHAs we have for this repo: {len(pr_shas)}")
            
            matched_introducing = introducing_shas & pr_shas
            matched_fixing = fixing_shas & pr_shas
            print(f"  Bug-introducing SHAs matching a PR: {len(matched_introducing)} / {len(introducing_shas)}")
            print(f"  Fixing SHAs matching a PR: {len(matched_fixing)} / {len(fixing_shas)}")
            print(f"\n  ⚠️  SHAs NOT matching any PR commit:")
            print(f"     Introducing: {len(introducing_shas - pr_shas)} unmatched")
            print(f"     Fixing: {len(fixing_shas - pr_shas)} unmatched")
        except Exception as e:
            print(f"  (Could not load HF pr_commits: {e})")
        
        # Check merge commit SHAs
        merge_meta_path = BASE_DIR / "data" / "processed" / "pr_merge_metadata.parquet"
        if merge_meta_path.exists():
            merge_meta = pd.read_parquet(str(merge_meta_path))
            repo_merge = merge_meta[merge_meta["pr_id"].isin(pr_ids)]
            merge_shas = set(repo_merge["merge_commit_sha"].dropna())
            print(f"\n  Merge commit SHAs for this repo: {len(merge_shas)}")
            matched_intro_merge = introducing_shas & merge_shas
            matched_fix_merge = fixing_shas & merge_shas
            print(f"  Bug-introducing matching merge SHA: {len(matched_intro_merge)}")
            print(f"  Fixing matching merge SHA: {len(matched_fix_merge)}")

# ── 5. szz_observable analysis ──
print("\n" + "=" * 70)
print("STAGE 5: SZZ_OBSERVABLE ANALYSIS")
print("=" * 70)

print(f"\nTotal PRs: {len(df)}")
print(f"Merged PRs: {df['merged_at'].notna().sum()}")
print(f"szz_observable=1: {(df['szz_observable'] == 1).sum()}")
print(f"szz_observable=0 but merged: {((df['szz_observable'] == 0) & (df['merged_at'].notna())).sum()}")
print(f"\nszz_observable by ai_pr:")
print(df.groupby("ai_pr")["szz_observable"].value_counts().unstack(fill_value=0))

# Check if szz_observable is set properly for repos that were processed
processed_repos_with_results = set()
for d in result_dirs:
    slug = d.name
    for r in df["repo_full_name"].unique():
        if safe_repo_slug(r) == slug:
            processed_repos_with_results.add(r)

print(f"\nRepos with SZZ result dirs: {len(processed_repos_with_results)}")
for_processed = df[df["repo_full_name"].isin(processed_repos_with_results)]
print(f"PRs in repos with SZZ results: {len(for_processed)}")
print(f"  szz_observable=1: {(for_processed['szz_observable'] == 1).sum()}")
print(f"  szz_observable=0: {(for_processed['szz_observable'] == 0).sum()}")
print(f"  Merged but not observable: {((for_processed['szz_observable'] == 0) & (for_processed['merged_at'].notna())).sum()}")

# ── 6. Summary of repos NOT processed ──
all_repos = set(df["repo_full_name"].unique())
not_processed = all_repos - processed_repos_with_results
print(f"\nRepos WITHOUT any SZZ result dir: {len(not_processed)}")
not_processed_df = df[df["repo_full_name"].isin(not_processed)]
print(f"  PRs in unprocessed repos: {len(not_processed_df)}")
print(f"  Of which merged: {not_processed_df['merged_at'].notna().sum()}")
print(f"  Of which candidate fix: {(not_processed_df['szz_candidate_fix'] == 1).sum()}")
