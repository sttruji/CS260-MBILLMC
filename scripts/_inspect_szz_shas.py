#!/usr/bin/env python3
"""Inspect SZZ output format to diagnose SHA mismatch."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inspect_repo(repo_key):
    result_path = os.path.join(ROOT, "data", "cache", "szz_results", repo_key, "results", "fix_and_introducers_pairs.json")
    issue_path = os.path.join(ROOT, "data", "cache", "szz_issue_lists", f"{repo_key}_issue_list.json")

    print(f"=== {repo_key} ===")

    # Load SZZ output
    with open(result_path) as f:
        szz_data = json.load(f)

    print(f"SZZ output: {len(szz_data)} keys")
    first_key = list(szz_data.keys())[0]
    first_val = szz_data[first_key]
    print(f"  Key type sample: {repr(first_key[:20])}... (len={len(first_key)})")
    print(f"  Value type: {type(first_val).__name__}")
    if isinstance(first_val, list):
        print(f"  Value sample: {[v[:20] for v in first_val[:3]]}")
    else:
        print(f"  Value sample: {repr(str(first_val)[:60])}")

    # Show first 3 full entries
    print("\n  First 3 entries:")
    for i, (k, v) in enumerate(list(szz_data.items())[:3]):
        print(f"    [{i}] bug_sha={k}")
        if isinstance(v, list):
            for vs in v[:3]:
                print(f"         fix_sha={vs}")
            if len(v) > 3:
                print(f"         ...and {len(v)-3} more")
        else:
            print(f"         fix_sha(s)={v}")

    # Collect all SHAs from SZZ output
    all_bug_shas = set(szz_data.keys())
    all_fix_shas = set()
    total_pairs = 0
    for val in szz_data.values():
        if isinstance(val, list):
            all_fix_shas.update(val)
            total_pairs += len(val)
        else:
            all_fix_shas.add(str(val))
            total_pairs += 1

    print(f"\n  Total pairs: {total_pairs}")
    print(f"  Unique bug SHAs: {len(all_bug_shas)}")
    print(f"  Unique fix SHAs: {len(all_fix_shas)}")

    # Load issue list
    with open(issue_path) as f:
        issue_data = json.load(f)

    print(f"\n  Issue list: {len(issue_data)} entries")
    print(f"  First entry keys: {list(issue_data[0].keys())}")
    for entry in issue_data[:3]:
        print(f"    {json.dumps(entry)[:200]}")

    # Extract SHAs from issue list
    issue_shas = set()
    for entry in issue_data:
        for k, v in entry.items():
            if isinstance(v, str) and len(v) >= 7:
                issue_shas.add(v)

    print(f"\n  All string values from issue list: {len(issue_shas)}")

    # Check overlaps
    fix_overlap = issue_shas & all_fix_shas
    bug_overlap = issue_shas & all_bug_shas
    print(f"  Issue SHAs matching fix SHAs: {len(fix_overlap)}")
    print(f"  Issue SHAs matching bug SHAs: {len(bug_overlap)}")

    # Check SHA lengths
    fix_sha_lens = set(len(s) for s in all_fix_shas)
    bug_sha_lens = set(len(s) for s in all_bug_shas)
    issue_sha_lens = set(len(s) for s in issue_shas)
    print(f"\n  Fix SHA lengths: {fix_sha_lens}")
    print(f"  Bug SHA lengths: {bug_sha_lens}")
    print(f"  Issue SHA lengths: {issue_sha_lens}")

    # Check if short SHAs are prefixes of full SHAs
    if fix_sha_lens != issue_sha_lens:
        print("\n  *** SHA LENGTH MISMATCH DETECTED ***")
        sample_fix = list(all_fix_shas)[:3]
        sample_issue = list(issue_shas)[:3]
        print(f"  Sample fix SHAs: {sample_fix}")
        print(f"  Sample issue SHAs: {sample_issue}")

        # Try prefix matching
        prefix_matches = 0
        for isha in list(issue_shas)[:20]:
            for fsha in all_fix_shas:
                if fsha.startswith(isha) or isha.startswith(fsha):
                    prefix_matches += 1
                    print(f"    Prefix match: issue={isha} ~ fix={fsha}")
                    break
        print(f"  Prefix matches (first 20 issue SHAs): {prefix_matches}")


# Check several repos
for repo_key in ["dotnet__sdk", "crewAIInc__crewAI", "julep-ai__julep"]:
    result_path = os.path.join(ROOT, "data", "cache", "szz_results", repo_key, "results", "fix_and_introducers_pairs.json")
    if os.path.exists(result_path):
        inspect_repo(repo_key)
        print("\n" + "="*70 + "\n")
