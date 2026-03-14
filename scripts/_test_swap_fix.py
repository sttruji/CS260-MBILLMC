#!/usr/bin/env python3
"""Quick test: re-run pair building with the swapped order and count matches."""
import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from rq1_v2 import load_szz_pairs
from pathlib import Path

# Test on the example data first
example_path = Path(ROOT) / "data/cache/tools/SZZUnleashed/examples/data/fix_and_introducers_pairs.json"
pairs = load_szz_pairs(example_path)
print(f"Example data: {len(pairs)} pairs loaded")
if pairs:
    print(f"  First pair: bug_introducing={pairs[0][0][:12]}..., fixing={pairs[0][1][:12]}...")

# Load the example issue list and verify fixing SHAs match
with open(ROOT + "/data/cache/tools/SZZUnleashed/examples/data/issue_list.json") as f:
    issue_data = json.load(f)
fix_hashes = set(v["hash"] for v in issue_data.values() if isinstance(v, dict) and "hash" in v)

fixing_shas_from_pairs = set(p[1] for p in pairs)  # element [1] = fixing after swap
bug_shas_from_pairs = set(p[0] for p in pairs)      # element [0] = bug-introducing after swap

print(f"\n  Fix hashes from issue list: {len(fix_hashes)}")
print(f"  Fixing SHAs from pairs (should match issue list): {len(fixing_shas_from_pairs & fix_hashes)} / {len(fixing_shas_from_pairs)}")
print(f"  Bug SHAs from pairs (should NOT match): {len(bug_shas_from_pairs & fix_hashes)} / {len(bug_shas_from_pairs)}")

# Now test with real data - dotnet/sdk
print("\n\n=== REAL DATA: dotnet__sdk ===")
real_path = Path(ROOT) / "data/cache/szz_results/dotnet__sdk/results/fix_and_introducers_pairs.json"
real_pairs = load_szz_pairs(real_path)
print(f"Loaded {len(real_pairs)} pairs")

# Load our issue list
with open(ROOT + "/data/cache/szz_issue_lists/dotnet__sdk_issue_list.json") as f:
    our_issues = json.load(f)
our_fix_hashes = set()
for k, v in our_issues.items():
    if isinstance(v, dict) and "hash" in v:
        our_fix_hashes.add(v["hash"])
print(f"Fix hashes from our issue list: {len(our_fix_hashes)}")

fixing_from_real = set(p[1] for p in real_pairs)
bug_from_real = set(p[0] for p in real_pairs)

print(f"Unique fixing SHAs (after swap): {len(fixing_from_real)}")
print(f"Unique bug-introducing SHAs (after swap): {len(bug_from_real)}")
print(f"Fixing matching our issue list: {len(our_fix_hashes & fixing_from_real)} / {len(fixing_from_real)}")
print(f"Bug-introducing matching our issue list: {len(our_fix_hashes & bug_from_real)} / {len(bug_from_real)}")

# Now check across ALL repos to estimate the new match count
print("\n\n=== ESTIMATING NEW MATCH RATE ACROSS ALL REPOS ===")
import pandas as pd

# Load the main frame
main_frame = pd.read_parquet(os.path.join(ROOT, "results", "rq1_main_frame_v2.parquet"))
szz_pairs_df = pd.read_parquet(os.path.join(ROOT, "results", "rq1_szz_pairs.parquet"))

szz_results_dir = Path(ROOT) / "data/cache/szz_results"
total_raw = 0
total_fix_match = 0
total_bug_match = 0
repos_with_matches = 0

for repo_dir in sorted(szz_results_dir.iterdir()):
    if not repo_dir.is_dir():
        continue
    pair_file = repo_dir / "results" / "fix_and_introducers_pairs.json"
    if not pair_file.exists():
        continue
    
    repo_key = repo_dir.name
    repo_name = repo_key.replace("__", "/")
    
    # Load pairs with new (corrected) order
    pairs = load_szz_pairs(pair_file)
    if not pairs:
        continue
    
    total_raw += len(pairs)
    
    # Get PR commit SHAs for this repo from the issue list
    issue_file = Path(ROOT) / "data/cache/szz_issue_lists" / f"{repo_key}_issue_list.json"
    if issue_file.exists():
        with open(issue_file) as f:
            issues = json.load(f)
        issue_hashes = set()
        if isinstance(issues, dict):
            for v in issues.values():
                if isinstance(v, dict) and "hash" in v:
                    issue_hashes.add(v["hash"])
        
        fixing_shas = set(p[1] for p in pairs)
        bug_shas = set(p[0] for p in pairs)
        
        fix_matches = len(issue_hashes & fixing_shas)
        bug_matches = len(issue_hashes & bug_shas)
        
        if fix_matches > 0:
            total_fix_match += fix_matches
            repos_with_matches += 1
            if repos_with_matches <= 30:
                print(f"  {repo_name}: {len(pairs)} pairs, {fix_matches}/{len(fixing_shas)} fixing SHAs match issue list, {bug_matches}/{len(bug_shas)} bug SHAs match")

print(f"\nTotal raw pairs: {total_raw}")
print(f"Repos with fixing SHA matches: {repos_with_matches}")
print(f"Total fixing SHA matches: {total_fix_match}")
