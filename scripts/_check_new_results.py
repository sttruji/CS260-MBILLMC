"""Quick check of new pipeline results after pair-order fix."""
import pandas as pd

df = pd.read_parquet("results/rq1_main_frame_v2.parquet")
df["is_ai"] = df["ai_pr"].astype(int)

print("=== NEW RESULTS (POST-FIX) ===")
print(f"Total PRs: {len(df)}")

n_def = int(df["has_defect_90d"].sum())
ai_def = int(((df["has_defect_90d"] == 1) & (df["is_ai"] == 1)).sum())
hu_def = int(((df["has_defect_90d"] == 1) & (df["is_ai"] == 0)).sum())
print(f"Defective (90d): {n_def}  (AI={ai_def}, Human={hu_def})")
print()

for lbl, g in df.groupby("is_ai"):
    nd = int((g["has_defect_90d"] == 1).sum())
    tag = "AI" if lbl else "Human"
    rate = nd / len(g) * 100
    print(f"  {tag}: {nd}/{len(g)} = {rate:.2f}%")

print()
obs = df[df["szz_observable"] == 1]
print(f"Observable: {len(obs)}")
for lbl, g in obs.groupby("is_ai"):
    nd = int((g["has_defect_90d"] == 1).sum())
    tag = "AI" if lbl else "Human"
    rate = nd / len(g) * 100
    print(f"  {tag} obs: {nd}/{len(g)} = {rate:.2f}%")

print()
pairs = pd.read_parquet("results/rq1_szz_pairs.parquet")
print(f"SZZ pairs: {len(pairs)}")
repos_w_defects = df[df["has_defect_90d"] == 1]["repo_full_name"].nunique()
print(f"Repos w/ defects: {repos_w_defects}")

for w in ["30d", "60d", "90d"]:
    c = f"has_defect_{w}"
    print(f"  {w}: {int(df[c].sum())} defective")
