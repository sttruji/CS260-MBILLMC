"""
Comparative Analysis Script: Productivity vs Maintenance Metrics
===============================================================
Goal: Determine if AI's "productivity boost" correlates with higher maintenance debt.

Metrics:
- Productivity (Speed, Volume):
  - Speed: Time from PR creation to merge (days)
  - Volume: NLOC total, additions, commit count

- Maintenance (Rework, Churn):
  - Churn: Lines re-modified within 90 days (churn_lines_90d)
  - Rework: Number of rework events (rework_events)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DATA = DATA_DIR / "processed_data.csv"
REWORK_METRICS = DATA_DIR / "rework_metrics.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the datasets."""
    # Load processed PR data
    pr_df = pd.read_csv(PROCESSED_DATA)
    
    # Load rework/churn metrics
    rework_df = pd.read_csv(REWORK_METRICS)
    
    # Convert timestamps
    pr_df['created_at'] = pd.to_datetime(pr_df['created_at'], utc=True)
    pr_df['merged_at'] = pd.to_datetime(pr_df['merged_at'], utc=True)
    
    return pr_df, rework_df


def merge_datasets(pr_df: pd.DataFrame, rework_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PR data with rework metrics on repo_name and pr_number."""
    merged = pr_df.merge(
        rework_df[['repo_name', 'pr_number', 'churn_lines_90d', 'rework_events', 'files_reworked_count']],
        on=['repo_name', 'pr_number'],
        how='left'
    )
    
    # Fill missing rework metrics with 0 (no rework detected)
    merged['churn_lines_90d'] = merged['churn_lines_90d'].fillna(0)
    merged['rework_events'] = merged['rework_events'].fillna(0)
    merged['files_reworked_count'] = merged['files_reworked_count'].fillna(0)
    
    return merged


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute productivity and maintenance metrics."""
    # Speed: Time to merge (in days)
    df['time_to_merge_days'] = (df['merged_at'] - df['created_at']).dt.total_seconds() / (24 * 3600)
    
    # Normalize churn by volume (churn rate)
    df['churn_rate'] = df['churn_lines_90d'] / (df['nloc_total'].replace(0, 1))
    
    # Rework rate (rework events per file changed)
    df['rework_rate'] = df['rework_events'] / (df['changed_files'].replace(0, 1))
    
    return df


def statistical_comparison(ai_data: pd.Series, human_data: pd.Series, metric_name: str) -> dict:
    """Perform statistical comparison between AI and Human groups."""
    # Remove NaN values
    ai_clean = ai_data.dropna()
    human_clean = human_data.dropna()
    
    # Descriptive statistics
    ai_stats = {
        'n': len(ai_clean),
        'mean': ai_clean.mean(),
        'median': ai_clean.median(),
        'std': ai_clean.std(),
        'min': ai_clean.min(),
        'max': ai_clean.max()
    }
    
    human_stats = {
        'n': len(human_clean),
        'mean': human_clean.mean(),
        'median': human_clean.median(),
        'std': human_clean.std(),
        'min': human_clean.min(),
        'max': human_clean.max()
    }
    
    # Statistical tests
    # Mann-Whitney U test (non-parametric, robust to non-normal distributions)
    if len(ai_clean) > 0 and len(human_clean) > 0:
        u_stat, u_pvalue = stats.mannwhitneyu(ai_clean, human_clean, alternative='two-sided')
        
        # Independent samples t-test
        t_stat, t_pvalue = stats.ttest_ind(ai_clean, human_clean, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((ai_clean.std()**2 + human_clean.std()**2) / 2)
        if pooled_std > 0:
            cohens_d = (ai_clean.mean() - human_clean.mean()) / pooled_std
        else:
            cohens_d = 0
    else:
        u_stat, u_pvalue = np.nan, np.nan
        t_stat, t_pvalue = np.nan, np.nan
        cohens_d = np.nan
    
    return {
        'metric': metric_name,
        'ai': ai_stats,
        'human': human_stats,
        'mann_whitney_u': {'statistic': u_stat, 'p_value': u_pvalue},
        't_test': {'statistic': t_stat, 'p_value': t_pvalue},
        'cohens_d': cohens_d
    }


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Analyze correlations between productivity and maintenance metrics."""
    productivity_cols = ['time_to_merge_days', 'nloc_total', 'additions', 'commit_count']
    maintenance_cols = ['churn_lines_90d', 'rework_events', 'churn_rate', 'rework_rate']
    
    correlations = {}
    
    for prod_col in productivity_cols:
        for maint_col in maintenance_cols:
            # Filter valid data
            valid_mask = df[prod_col].notna() & df[maint_col].notna()
            if valid_mask.sum() < 3:
                continue
                
            prod_data = df.loc[valid_mask, prod_col]
            maint_data = df.loc[valid_mask, maint_col]
            
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(prod_data, maint_data)
            
            # Spearman correlation (rank-based, robust to outliers)
            spearman_r, spearman_p = stats.spearmanr(prod_data, maint_data)
            
            correlations[f'{prod_col}_vs_{maint_col}'] = {
                'pearson': {'r': pearson_r, 'p_value': pearson_p},
                'spearman': {'rho': spearman_r, 'p_value': spearman_p},
                'n': valid_mask.sum()
            }
    
    return correlations


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_comparison(result: dict):
    """Print comparison results in a readable format."""
    metric = result['metric']
    ai = result['ai']
    human = result['human']
    
    print(f"\n{metric}:")
    print(f"  {'':20} {'AI':>15} {'Human':>15}")
    print(f"  {'-'*50}")
    print(f"  {'N':20} {ai['n']:>15} {human['n']:>15}")
    print(f"  {'Mean':20} {ai['mean']:>15.2f} {human['mean']:>15.2f}")
    print(f"  {'Median':20} {ai['median']:>15.2f} {human['median']:>15.2f}")
    print(f"  {'Std Dev':20} {ai['std']:>15.2f} {human['std']:>15.2f}")
    print(f"  {'Min':20} {ai['min']:>15.2f} {human['min']:>15.2f}")
    print(f"  {'Max':20} {ai['max']:>15.2f} {human['max']:>15.2f}")
    
    # Statistical significance
    p_value = result['mann_whitney_u']['p_value']
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"\n  Mann-Whitney U p-value: {p_value:.4f} {significance}")
    print(f"  Effect Size (Cohen's d): {result['cohens_d']:.3f}")
    
    # Interpretation
    d = abs(result['cohens_d'])
    if d < 0.2:
        effect_interp = "negligible"
    elif d < 0.5:
        effect_interp = "small"
    elif d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Effect interpretation: {effect_interp}")


def main():
    print("=" * 60)
    print(" COMPARATIVE ANALYSIS: AI vs Human Code Contributions")
    print(" Productivity (Speed, Volume) vs Maintenance (Rework, Churn)")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading data...")
    pr_df, rework_df = load_data()
    print(f"  Loaded {len(pr_df)} PRs and {len(rework_df)} rework records")
    
    # Merge datasets
    df = merge_datasets(pr_df, rework_df)
    print(f"  Merged dataset: {len(df)} records")
    
    # Compute metrics
    df = compute_metrics(df)
    
    # Split by classification
    ai_df = df[df['classification'] == 'AI']
    human_df = df[df['classification'] == 'Human']
    
    print(f"\n  AI PRs: {len(ai_df)}")
    print(f"  Human PRs: {len(human_df)}")
    
    # =====================================================================
    # PRODUCTIVITY METRICS COMPARISON
    # =====================================================================
    print_section("PRODUCTIVITY METRICS (Speed & Volume)")
    
    productivity_metrics = [
        ('time_to_merge_days', 'Time to Merge (days)'),
        ('nloc_total', 'Total NLOC'),
        ('additions', 'Lines Added'),
        ('commit_count', 'Commit Count'),
        ('avg_nloc_per_commit', 'Avg NLOC per Commit')
    ]
    
    productivity_results = []
    for col, name in productivity_metrics:
        result = statistical_comparison(ai_df[col], human_df[col], name)
        productivity_results.append(result)
        print_comparison(result)
    
    # =====================================================================
    # MAINTENANCE METRICS COMPARISON
    # =====================================================================
    print_section("MAINTENANCE METRICS (Rework & Churn)")
    
    maintenance_metrics = [
        ('churn_lines_90d', 'Churn Lines (90 days)'),
        ('rework_events', 'Rework Events'),
        ('files_reworked_count', 'Files Reworked'),
        ('churn_rate', 'Churn Rate (churn/NLOC)'),
        ('rework_rate', 'Rework Rate (events/files)')
    ]
    
    maintenance_results = []
    for col, name in maintenance_metrics:
        result = statistical_comparison(ai_df[col], human_df[col], name)
        maintenance_results.append(result)
        print_comparison(result)
    
    # =====================================================================
    # CORRELATION ANALYSIS
    # =====================================================================
    print_section("CORRELATION ANALYSIS: Productivity vs Maintenance")
    
    print("\nDoes higher productivity correlate with higher maintenance debt?")
    print("(Looking for positive correlations between productivity and churn/rework)")
    
    # Overall correlations
    print("\n--- Overall Dataset ---")
    correlations = correlation_analysis(df)
    
    key_correlations = [
        ('nloc_total_vs_churn_lines_90d', 'Volume (NLOC) vs Churn'),
        ('additions_vs_churn_lines_90d', 'Additions vs Churn'),
        ('commit_count_vs_rework_events', 'Commit Count vs Rework'),
        ('time_to_merge_days_vs_churn_lines_90d', 'Speed vs Churn'),
    ]
    
    for key, label in key_correlations:
        if key in correlations:
            corr = correlations[key]
            r = corr['spearman']['rho']
            p = corr['spearman']['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label}: ρ = {r:.3f}, p = {p:.4f} {sig}")
    
    # Correlations by group
    print("\n--- AI PRs Only ---")
    ai_correlations = correlation_analysis(ai_df)
    for key, label in key_correlations:
        if key in ai_correlations:
            corr = ai_correlations[key]
            r = corr['spearman']['rho']
            p = corr['spearman']['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label}: ρ = {r:.3f}, p = {p:.4f} {sig}")
    
    print("\n--- Human PRs Only ---")
    human_correlations = correlation_analysis(human_df)
    for key, label in key_correlations:
        if key in human_correlations:
            corr = human_correlations[key]
            r = corr['spearman']['rho']
            p = corr['spearman']['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label}: ρ = {r:.3f}, p = {p:.4f} {sig}")
    
    # =====================================================================
    # SUMMARY & CONCLUSIONS
    # =====================================================================
    print_section("SUMMARY & KEY FINDINGS")
    
    # Productivity summary
    print("\n📊 PRODUCTIVITY (AI vs Human):")
    for result in productivity_results:
        ai_mean = result['ai']['mean']
        human_mean = result['human']['mean']
        diff_pct = ((ai_mean - human_mean) / human_mean * 100) if human_mean != 0 else 0
        direction = "higher" if ai_mean > human_mean else "lower"
        p = result['mann_whitney_u']['p_value']
        sig_marker = " (significant)" if p < 0.05 else ""
        print(f"  • {result['metric']}: AI is {abs(diff_pct):.1f}% {direction}{sig_marker}")
    
    # Maintenance summary
    print("\n🔧 MAINTENANCE DEBT (AI vs Human):")
    for result in maintenance_results:
        ai_mean = result['ai']['mean']
        human_mean = result['human']['mean']
        diff_pct = ((ai_mean - human_mean) / human_mean * 100) if human_mean != 0 else 0
        direction = "higher" if ai_mean > human_mean else "lower"
        p = result['mann_whitney_u']['p_value']
        sig_marker = " (significant)" if p < 0.05 else ""
        print(f"  • {result['metric']}: AI is {abs(diff_pct):.1f}% {direction}{sig_marker}")
    
    # Key insight
    print("\n" + "="*60)
    print(" KEY INSIGHT: Does AI Productivity Come at the Cost of")
    print(" Higher Maintenance Debt?")
    print("="*60)
    
    # Analyze the tradeoff
    ai_nloc_mean = ai_df['nloc_total'].mean()
    human_nloc_mean = human_df['nloc_total'].mean()
    ai_churn_mean = ai_df['churn_lines_90d'].mean()
    human_churn_mean = human_df['churn_lines_90d'].mean()
    
    productivity_boost = (ai_nloc_mean / human_nloc_mean) if human_nloc_mean > 0 else 1
    maintenance_diff = (ai_churn_mean / human_churn_mean) if human_churn_mean > 0 else 1
    
    print(f"\n  Productivity ratio (AI/Human NLOC): {productivity_boost:.2f}x")
    print(f"  Maintenance ratio (AI/Human Churn): {maintenance_diff:.2f}x")
    
    if productivity_boost > 1 and maintenance_diff > 1:
        print("\n  ⚠️  FINDING: AI code shows BOTH higher productivity AND higher")
        print("     maintenance debt. The productivity boost may come at a cost.")
    elif productivity_boost > 1 and maintenance_diff <= 1:
        print("\n  ✅ FINDING: AI code shows higher productivity WITHOUT")
        print("     increased maintenance debt. Win-win scenario!")
    elif productivity_boost <= 1 and maintenance_diff > 1:
        print("\n  ❌ FINDING: AI code shows LOWER productivity with HIGHER")
        print("     maintenance debt. Concerning pattern.")
    else:
        print("\n  ℹ️  FINDING: Both productivity and maintenance are comparable")
        print("     between AI and Human contributions.")
    
    print("\n" + "="*60)
    print(" Note: Statistical significance (p < 0.05) marked with *")
    print(" Effect sizes: negligible (<0.2), small (0.2-0.5),")
    print("               medium (0.5-0.8), large (>0.8)")
    print("="*60)


if __name__ == "__main__":
    main()
