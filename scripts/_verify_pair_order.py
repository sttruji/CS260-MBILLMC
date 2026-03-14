#!/usr/bin/env python3
"""Verify the pair order in SZZUnleashed output."""
import json

# Load example issue list
with open("data/cache/tools/SZZUnleashed/examples/data/issue_list.json") as f:
    issue_data = json.load(f)

print(f"Issue list type: {type(issue_data).__name__}, length: {len(issue_data)}")
for i, (k, v) in enumerate(list(issue_data.items())[:3]):
    print(f"  key={k}, value={v}")

# Extract fix SHAs from issue list
fix_hashes = set()
for k, v in issue_data.items():
    if isinstance(v, dict):
        if "hash" in v:
            fix_hashes.add(v["hash"])
    elif isinstance(v, str) and len(v) >= 7:
        fix_hashes.add(v)
    if isinstance(v, list):
        for item in v:
            if isinstance(item, str) and len(item) >= 7:
                fix_hashes.add(item)
            elif isinstance(item, dict) and "hash" in item:
                fix_hashes.add(item["hash"])

print(f"\nFix hashes from issue list: {len(fix_hashes)}")
print(f"Sample: {list(fix_hashes)[:3]}")

# Load SZZ output
with open("data/cache/tools/SZZUnleashed/examples/data/fix_and_introducers_pairs.json") as f:
    pairs = json.load(f)

elem0_set = set(p[0] for p in pairs)
elem1_set = set(p[1] for p in pairs)

print(f"\nSZZ output pairs: {len(pairs)}")
print(f"Unique element[0]: {len(elem0_set)}")
print(f"Unique element[1]: {len(elem1_set)}")
print(f"\nelement[0] matching fix hashes: {len(fix_hashes & elem0_set)}")
print(f"element[1] matching fix hashes: {len(fix_hashes & elem1_set)}")

print(f"\n=== CONCLUSION ===")
if len(fix_hashes & elem0_set) > len(fix_hashes & elem1_set):
    print("element[0] = FIXING commit, element[1] = BUG-INTRODUCING commit")
    print("=> Our code has the order SWAPPED!")
else:
    print("element[0] = BUG-INTRODUCING commit, element[1] = FIXING commit")
    print("=> Our code has the correct order")

# Now verify with our actual data
print("\n\n=== VERIFYING WITH REAL DATA (dotnet__sdk) ===")
with open("data/cache/szz_results/dotnet__sdk/results/fix_and_introducers_pairs.json") as f:
    szz_pairs = json.load(f)

with open("data/cache/szz_issue_lists/dotnet__sdk_issue_list.json") as f:
    issue_list = json.load(f)

# Extract hashes from our issue list
our_fix_hashes = set()
for entry in issue_list:
    if isinstance(entry, dict):
        for k, v in entry.items():
            if isinstance(v, str) and len(v) >= 20:
                our_fix_hashes.add(v)

print(f"Fix hashes from our issue list: {len(our_fix_hashes)}")
print(f"Sample: {list(our_fix_hashes)[:3]}")

e0 = set(p[0] for p in szz_pairs)
e1 = set(p[1] for p in szz_pairs)

print(f"\nelement[0] matching our fix hashes: {len(our_fix_hashes & e0)}")
print(f"element[1] matching our fix hashes: {len(our_fix_hashes & e1)}")

if len(our_fix_hashes & e0) > len(our_fix_hashes & e1):
    print("\n*** CONFIRMED: element[0] = FIXING, element[1] = BUG-INTRODUCING ***")
    print("*** extract_pair_values() has the mapping BACKWARDS! ***")
