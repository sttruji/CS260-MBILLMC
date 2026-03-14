import pandas as pd

df = pd.read_parquet('results/rq1_main_frame_v2.parquet')
print('Shape:', df.shape)
print()
print('Columns:')
for c in sorted(df.columns):
    print(f'  {c}: {df[c].dtype}  nunique={df[c].nunique()}')
print()
print('language_type_group value_counts:')
print(df['language_type_group'].value_counts().sort_index())
print()
if 'primary_language' in df.columns:
    print('primary_language top 20:')
    print(df['primary_language'].value_counts().head(20))
print()
print('Defect rates by language_type_group x ai_pr:')
ct = df.groupby(['language_type_group', 'ai_pr']).agg(
    n=('has_defect_90d', 'count'),
    defects=('has_defect_90d', 'sum'),
    rate=('has_defect_90d', 'mean')
).reset_index()
print(ct.to_string())
print()
print('Sample repos per language_type_group:')
for g in sorted(df['language_type_group'].unique()):
    sub = df[df['language_type_group'] == g]
    if 'primary_language' in sub.columns:
        langs = sub['primary_language'].value_counts().head(5)
        print(f'  Group {g}: n={len(sub)}')
        print(langs.to_string())
        print()

# Also check domain_type_group and task_type_group
print('domain_type_group value_counts:')
print(df['domain_type_group'].value_counts().sort_index())
print()
print('task_type_group value_counts:')
print(df['task_type_group'].value_counts().sort_index())
print()

# observable subset
obs = df[df['szz_observable'] == True]
print(f'Observable subset: {len(obs)} rows')
ct2 = obs.groupby(['language_type_group', 'ai_pr']).agg(
    n=('has_defect_90d', 'count'),
    defects=('has_defect_90d', 'sum'),
    rate=('has_defect_90d', 'mean')
).reset_index()
print(ct2.to_string())
