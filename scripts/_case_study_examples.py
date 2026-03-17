import pandas as pd
import numpy as np

# Load the main results
df = pd.read_csv('results/rq1_main_frame_v2.csv')

# Find examples with defects
defect_prs = df[df['has_defect_90d'] == True].copy()

print('=' * 70)
print('RQ1 CASE STUDY: COMPARABLE AI vs Human PRs')
print('=' * 70)
print()

# Get AI and Human defect PRs
ai_defects = defect_prs[defect_prs['ai_pr'] == True].copy()
human_defects = defect_prs[defect_prs['ai_pr'] == False].copy()

# Find comparable pairs: same task type and similar PR size
print('Finding comparable AI vs Human PR pairs...')
print()

best_match = None
best_score = float('inf')

for _, ai_pr in ai_defects.iterrows():
    # Find human PRs with same task type
    same_task = human_defects[human_defects['task_type'] == ai_pr['task_type']]
    
    if len(same_task) == 0:
        continue
    
    for _, human_pr in same_task.iterrows():
        # Calculate similarity score based on PR size
        size_diff = abs(ai_pr['pr_size_loc'] - human_pr['pr_size_loc'])
        size_ratio = max(ai_pr['pr_size_loc'], human_pr['pr_size_loc']) / max(1, min(ai_pr['pr_size_loc'], human_pr['pr_size_loc']))
        
        # Prefer PRs in reasonable size range (100-2000 LOC) with similar sizes
        if 100 <= ai_pr['pr_size_loc'] <= 2000 and 100 <= human_pr['pr_size_loc'] <= 2000:
            score = size_ratio
            if score < best_score:
                best_score = score
                best_match = (ai_pr, human_pr)

if best_match:
    ai_example, human_example = best_match
    
    print('=' * 70)
    print('MATCHED CASE STUDY: Same Task Type, Similar PR Size')
    print('=' * 70)
    print()
    print(f"Task Type: {ai_example['task_type']}")
    print(f"AI PR Size: {int(ai_example['pr_size_loc'])} LOC")
    print(f"Human PR Size: {int(human_example['pr_size_loc'])} LOC")
    print(f"Size Ratio: {best_score:.2f}x")
    print()
    
    print('-' * 70)
    print('AI-GENERATED PR')
    print('-' * 70)
    print(f"Repository:      {ai_example['repo_full_name']}")
    print(f"PR Number:       #{int(ai_example['number'])}")
    print(f"Title:           {str(ai_example['title'])[:70]}")
    print(f"AI Agent:        {ai_example['agent']}")
    print(f"Task Type:       {ai_example['task_type']}")
    print(f"PR Size:         {int(ai_example['pr_size_loc'])} LOC (+{int(ai_example['pr_additions'])} / -{int(ai_example['pr_deletions'])})")
    print(f"Defects (90d):   {int(ai_example['defect_count_90d'])}")
    print(f"Defect Density:  {ai_example['defect_density_90d']:.6f}")
    print(f"URL:             {ai_example['html_url']}")
    print()
    
    print('-' * 70)
    print('HUMAN-GENERATED PR')
    print('-' * 70)
    print(f"Repository:      {human_example['repo_full_name']}")
    print(f"PR Number:       #{int(human_example['number'])}")
    print(f"Title:           {str(human_example['title'])[:70]}")
    print(f"Author:          Human")
    print(f"Task Type:       {human_example['task_type']}")
    print(f"PR Size:         {int(human_example['pr_size_loc'])} LOC (+{int(human_example['pr_additions'])} / -{int(human_example['pr_deletions'])})")
    print(f"Defects (90d):   {int(human_example['defect_count_90d'])}")
    print(f"Defect Density:  {human_example['defect_density_90d']:.6f}")
    print(f"URL:             {human_example['html_url']}")
    print()
    
    print('=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)
    print(f"{'Metric':<25} {'AI PR':<20} {'Human PR':<20}")
    print('-' * 70)
    print(f"{'Task Type':<25} {ai_example['task_type']:<20} {human_example['task_type']:<20}")
    print(f"{'PR Size (LOC)':<25} {int(ai_example['pr_size_loc']):<20} {int(human_example['pr_size_loc']):<20}")
    print(f"{'Defects (90d)':<25} {int(ai_example['defect_count_90d']):<20} {int(human_example['defect_count_90d']):<20}")
    print(f"{'Defect Density':<25} {ai_example['defect_density_90d']:<20.6f} {human_example['defect_density_90d']:<20.6f}")
    
else:
    print("No comparable matches found.")

print()
print('=' * 70)
print('OVERALL STATISTICS (for context)')
print('=' * 70)
total_ai = len(df[df['ai_pr'] == True])
total_human = len(df[df['ai_pr'] == False])
print(f'Total AI PRs: {total_ai:,}')
print(f'Total Human PRs: {total_human:,}')
print(f'AI defect rate (90d): {len(ai_defects)/total_ai*100:.2f}%')
print(f'Human defect rate (90d): {len(human_defects)/total_human*100:.2f}%')
