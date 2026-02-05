"""Main data processing pipeline - Stage 1: PR Ingestion & Classification."""
import os
import pandas as pd
from datetime import datetime
from extractors.pr_classifier import PRClassifier
from extractors.github_utils import GitHubUtils

def load_processed_repos(output_csv):
    """Load set of already-processed repos to avoid re-processing."""
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        return set(df['repo_name'].unique())
    return set()

def append_pr_data(pr_data, output_csv):
    """Append PR data to CSV incrementally (write after each repo)."""
    df_new = pd.DataFrame(pr_data)
    os.makedirs('data/processed', exist_ok=True)
    
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv, index=False)
    else:
        df_new.to_csv(output_csv, index=False)

def process_stage_1(input_csv='data/raw/targeted_repos.csv', output_csv='data/processed/pr_details.csv'):
    """
    Stage 1: Extract all PRs from candidate repos and classify them.
    Supports resuming from interruptions.
    
    Args:
        input_csv (str): Path to targeted_repos.csv
        output_csv (str): Path to output PR details
    """
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found")
        return
    
    # Load candidate repos
    repos_df = pd.read_csv(input_csv)
    processed_repos = load_processed_repos(output_csv)
    remaining_repos = [r for r in repos_df['full_name'] if r not in processed_repos]
    
    print(f"Processing {len(repos_df)} candidate repositories...")
    print(f"  Already processed: {len(processed_repos)}")
    print(f"  Remaining: {len(remaining_repos)}\n")
    
    # Initialize tools
    classifier = PRClassifier()
    github_utils = GitHubUtils()
    
    # Process each repository
    for idx, repo_name in enumerate(remaining_repos, start=len(processed_repos)+1):
        print(f"[{idx}/{len(repos_df)}] Processing {repo_name}...")
        
        # Fetch repository
        repo = github_utils.get_repository(repo_name)
        if not repo:
            print(f"  ✗ Failed to fetch repository")
            continue
        
        # Fetch and classify all merged PRs
        pr_count = 0
        ai_count = 0
        pr_batch = []
        
        try:
            for pr in github_utils.get_merged_prs(repo):
                # Classify PR
                classification = classifier.classify_pr(pr)
                
                # Extract metadata
                metadata = github_utils.get_pr_metadata(pr)
                
                # Get timestamp model inference
                timestamp_info = classifier.extract_timestamp_model(pr.created_at)
                
                # Combine all data
                pr_data = {
                    'repo_name': repo_name,
                    'pr_number': metadata['pr_number'],
                    'pr_title': metadata['title'],
                    'classification': classification['classification'],
                    'confidence': classification['confidence'],
                    'detection_signals': ','.join(classification['signals']),
                    'created_at': metadata['created_at'].isoformat(),
                    'merged_at': metadata['merged_at'].isoformat(),
                    'model_era': timestamp_info['era'],
                    'approx_model': timestamp_info['approx_model'],
                    'author': metadata['author'],
                    'nloc': metadata['additions'] + metadata['deletions'],
                    'additions': metadata['additions'],
                    'deletions': metadata['deletions'],
                    'changed_files': metadata['changed_files'],
                    'commits': metadata['commits']
                }
                
                pr_batch.append(pr_data)
                pr_count += 1
                
                if classification['classification'] == 'AI':
                    ai_count += 1
        
        except Exception as e:
            print(f"  ✗ Error processing PRs: {e}")
            continue
        
        # Save batch after each repo (incremental save)
        if pr_batch:
            append_pr_data(pr_batch, output_csv)
        
        print(f"  ✓ Processed {pr_count} PRs ({ai_count} classified as AI)\n")
    
    # Summary
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"\n{'='*60}")
        print(f"✓ Stage 1 Complete!")
        print(f"  Total PRs extracted: {len(df)}")
        print(f"  AI-classified: {len(df[df['classification'] == 'AI'])}")
        print(f"  Human-classified: {len(df[df['classification'] == 'Human'])}")
        print(f"  Output: {output_csv}")
        print(f"{'='*60}")
    else:
        print("No PR data collected")

if __name__ == "__main__":
    process_stage_1()