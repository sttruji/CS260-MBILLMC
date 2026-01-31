import os 
import json
import time 
import requests 
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

def check_rate_limit():
    """Checks remaining API calls to avoid hitting limits mid-scrape."""
    response = requests.get("https://api.github.com/rate_limit", headers=HEADERS)
    return response.json()['resources']['core']['remaining']

def filter_repository(repo_full_name):
    """
    Applies proposal filters: 
    1. >=50 merged PRs in the last year.
    2. >=10 AI-labeled PRs in the last year.
    """
    one_year_ago = (datetime.now() - timedelta(days=365)).isoformat()
    
    # 1. Check Total Merged PRs
    # Search query: merged PRs in the last year for this repo
    pr_query = f"repo:{repo_full_name} is:pr is:merged merged:>{one_year_ago}"
    pr_url = f"https://api.github.com/search/issues?q={pr_query}&per_page=1"
    pr_resp = requests.get(pr_url, headers=HEADERS).json()
    total_merged = pr_resp.get('total_count', 0)
    
    if total_merged < 50:
        return False, 0, 0

    # 2. Check AI-Labeled PRs
    # Expanded list of common AI-related labels (50+)
    ai_labels = [
        '"AI-Generated"', '"AI Generated"', '"AI-generated"', '"AI generated"',
        '"copilot"', '"Copilot"', '"github-copilot"', '"GitHub Copilot"',
        '"gpt"', '"gpt-3"', '"gpt-4"', '"gpt-generated"', '"gpt generated"',
        '"gpt3"', '"gpt4"', '"gpt3-generated"', '"gpt4-generated"',
        '"openai"', '"OpenAI"', '"openai-generated"', '"openai generated"',
        '"chatgpt"', '"ChatGPT"', '"chatgpt-generated"', '"chatgpt generated"',
        '"llm"', '"LLM"', '"llm-generated"', '"llm generated"',
        '"ai-assist"', '"AI-Assist"', '"ai-assisted"', '"AI-Assisted"',
        '"ai"', '"AI"', '"ai commit"', '"AI commit"',
        '"machine-learning"', '"Machine Learning"', '"ml"', '"ML"',
        '"autogen"', '"auto-gen"', '"auto generated"', '"auto-generated"',
        '"codegen"', '"code-gen"', '"code generated"', '"code-generated"',
        '"ai-bot"', '"AI-Bot"', '"bot"', '"Bot"',
        '"ai-pr"', '"AI-PR"', '"ai-pr-generated"', '"AI-PR-Generated"',
        '"ai-pull"', '"AI-Pull"', '"ai-pull-request"', '"AI-Pull-Request"',
        '"ai-change"', '"AI-Change"', '"ai-changes"', '"AI-Changes"',
        '"ai-update"', '"AI-Update"', '"ai-updated"', '"AI-Updated"',
        '"ai-patch"', '"AI-Patch"', '"ai-patched"', '"AI-Patched"',
        '"ai-contrib"', '"AI-Contrib"', '"ai-contribution"', '"AI-Contribution"',
        '"ai-commit"', '"AI-Commit"', '"ai-committed"', '"AI-Committed"',
        '"ai-feature"', '"AI-Feature"', '"ai-features"', '"AI-Features"',
        '"ai-bugfix"', '"AI-Bugfix"', '"ai-bug-fix"', '"AI-Bug-Fix"',
        '"ai-fix"', '"AI-Fix"', '"ai-fixed"', '"AI-Fixed"',
        '"ai-enhancement"', '"AI-Enhancement"', '"ai-enhanced"', '"AI-Enhanced"',
        '"ai-refactor"', '"AI-Refactor"', '"ai-refactored"', '"AI-Refactored"',
        '"ai-review"', '"AI-Review"', '"ai-reviewed"', '"AI-Reviewed"',
        '"ai-test"', '"AI-Test"', '"ai-tested"', '"AI-Tested"',
        '"ai-doc"', '"AI-Doc"', '"ai-docs"', '"AI-Docs"',
        '"ai-documentation"', '"AI-Documentation"', '"ai-doc-generated"', '"AI-Doc-Generated"',
        '"ai-script"', '"AI-Script"', '"ai-scripted"', '"AI-Scripted"',
        '"ai-automation"', '"AI-Automation"', '"ai-automated"', '"AI-Automated"',
        '"ai-helper"', '"AI-Helper"', '"ai-help"', '"AI-Help"',
        '"ai-support"', '"AI-Support"', '"ai-supported"', '"AI-Supported"',
        '"ai-generated-content"', '"AI-Generated-Content"', '"ai-content"', '"AI-Content"',
        '"ai-translation"', '"AI-Translation"', '"ai-translated"', '"AI-Translated"',
        '"ai-suggestion"', '"AI-Suggestion"', '"ai-suggested"', '"AI-Suggested"',
        '"ai-merge"', '"AI-Merge"', '"ai-merged"', '"AI-Merged"',
        '"ai-commit-message"', '"AI-Commit-Message"', '"ai-message"', '"AI-Message"',
        '"ai-label"', '"AI-Label"', '"ai-labeled"', '"AI-Labeled"',
        '"ai-tag"', '"AI-Tag"', '"ai-tagged"', '"AI-Tagged"',
        '"ai-release"', '"AI-Release"', '"ai-released"', '"AI-Released"',
        '"ai-pipeline"', '"AI-Pipeline"', '"ai-pipelined"', '"AI-Pipelined"',
        '"ai-ci"', '"AI-CI"', '"ai-cd"', '"AI-CD"',
        '"ai-integration"', '"AI-Integration"', '"ai-integrated"', '"AI-Integrated"',
        '"ai-bot-commit"', '"AI-Bot-Commit"', '"ai-bot-pr"', '"AI-Bot-PR"',
        '"ai-bot-pull"', '"AI-Bot-Pull"', '"ai-bot-pull-request"', '"AI-Bot-Pull-Request"',
        '"ai-bot-change"', '"AI-Bot-Change"', '"ai-bot-changes"', '"AI-Bot-Changes"',
        '"ai-bot-update"', '"AI-Bot-Update"', '"ai-bot-updated"', '"AI-Bot-Updated"',
        '"ai-bot-patch"', '"AI-Bot-Patch"', '"ai-bot-patched"', '"AI-Bot-Patched"',
        '"ai-bot-contrib"', '"AI-Bot-Contrib"', '"ai-bot-contribution"', '"AI-Bot-Contribution"',
        '"ai-bot-commit"', '"AI-Bot-Commit"', '"ai-bot-committed"', '"AI-Bot-Committed"',
        '"ai-bot-feature"', '"AI-Bot-Feature"', '"ai-bot-features"', '"AI-Bot-Features"',
        '"ai-bot-bugfix"', '"AI-Bot-Bugfix"', '"ai-bot-bug-fix"', '"AI-Bot-Bug-Fix"',
        '"ai-bot-fix"', '"AI-Bot-Fix"', '"ai-bot-fixed"', '"AI-Bot-Fixed"',
        '"ai-bot-enhancement"', '"AI-Bot-Enhancement"', '"ai-bot-enhanced"', '"AI-Bot-Enhanced"',
        '"ai-bot-refactor"', '"AI-Bot-Refactor"', '"ai-bot-refactored"', '"AI-Bot-Refactored"',
        '"ai-bot-review"', '"AI-Bot-Review"', '"ai-bot-reviewed"', '"AI-Bot-Reviewed"',
        '"ai-bot-test"', '"AI-Bot-Test"', '"ai-bot-tested"', '"AI-Bot-Tested"',
        '"ai-bot-doc"', '"AI-Bot-Doc"', '"ai-bot-docs"', '"AI-Bot-Docs"',
        '"ai-bot-documentation"', '"AI-Bot-Documentation"', '"ai-bot-doc-generated"', '"AI-Bot-Doc-Generated"',
        '"ai-bot-script"', '"AI-Bot-Script"', '"ai-bot-scripted"', '"AI-Bot-Scripted"',
        '"ai-bot-automation"', '"AI-Bot-Automation"', '"ai-bot-automated"', '"AI-Bot-Automated"',
        '"ai-bot-helper"', '"AI-Bot-Helper"', '"ai-bot-help"', '"AI-Bot-Help"',
        '"ai-bot-support"', '"AI-Bot-Support"', '"ai-bot-supported"', '"AI-Bot-Supported"',
        '"ai-bot-generated-content"', '"AI-Bot-Generated-Content"', '"ai-bot-content"', '"AI-Bot-Content"',
        '"ai-bot-translation"', '"AI-Bot-Translation"', '"ai-bot-translated"', '"AI-Bot-Translated"',
        '"ai-bot-suggestion"', '"AI-Bot-Suggestion"', '"ai-bot-suggested"', '"AI-Bot-Suggested"',
        '"ai-bot-merge"', '"AI-Bot-Merge"', '"ai-bot-merged"', '"AI-Bot-Merged"',
        '"ai-bot-commit-message"', '"AI-Bot-Commit-Message"', '"ai-bot-message"', '"AI-Bot-Message"',
        '"ai-bot-label"', '"AI-Bot-Label"', '"ai-bot-labeled"', '"AI-Bot-Labeled"',
        '"ai-bot-tag"', '"AI-Bot-Tag"', '"ai-bot-tagged"', '"AI-Bot-Tagged"',
        '"ai-bot-release"', '"AI-Bot-Release"', '"ai-bot-released"', '"AI-Bot-Released"',
        '"ai-bot-pipeline"', '"AI-Bot-Pipeline"', '"ai-bot-pipelined"', '"AI-Bot-Pipelined"',
        '"ai-bot-ci"', '"AI-Bot-CI"', '"ai-bot-cd"', '"AI-Bot-CD"',
        '"ai-bot-integration"', '"AI-Bot-Integration"', '"ai-bot-integrated"', '"AI-Bot-Integrated"'
    ]
    total_ai = 0
    for label in ai_labels:
        ai_query = f"repo:{repo_full_name} is:pr label:{label} merged:>{one_year_ago}"
        ai_url = f"https://api.github.com/search/issues?q={ai_query}&per_page=1"
        ai_resp = requests.get(ai_url, headers=HEADERS).json()
        total_ai += ai_resp.get('total_count', 0)
        time.sleep(1) # Small delay to respect search rate limits
    if total_ai >= 10:
        return True, total_merged, total_ai
    return False, total_merged, total_ai

def main(target_count=20):
    start_time = time.time()
    discovered_repos = []
    
    # List of search queries to try sequentially
    search_queries = [
        # Microsoft and high-activity orgs, likely to have AI-generated code and moderation
        "org:microsoft is:public stars:>1000 pushed:>2025-01-01",
        "org:microsoft is:public language:python stars:>500 pushed:>2025-01-01",
        "org:microsoft is:public language:javascript stars:>500 pushed:>2025-01-01",
        "org:microsoft is:public language:java stars:>500 pushed:>2025-01-01",
        # Other large orgs with open source and likely AI code
        "org:google is:public stars:>1000 pushed:>2025-01-01",
        "org:facebook is:public stars:>1000 pushed:>2025-01-01",
        "org:openai is:public stars:>100 pushed:>2025-01-01",
        "org:aws is:public stars:>500 pushed:>2025-01-01",
        "org:pytorch is:public stars:>500 pushed:>2025-01-01",
        # General fallback for high-activity repos with AI in PRs/issues
        "language:python stars:>1000 in:pr,issue \"AI-generated\" pushed:>2025-01-01",
        "language:javascript stars:>1000 in:pr,issue \"copilot\" pushed:>2025-01-01",
        "language:java stars:>1000 in:pr,issue \"AI-assisted\" pushed:>2025-01-01",
        # Add more queries as needed
    ]

    print(f"Starting discovery for {target_count} repos...")

    for search_query in search_queries:
        search_url = f"https://api.github.com/search/repositories?q={search_query}&sort=updated"
        page = 1
        while len(discovered_repos) < target_count:
            response = requests.get(f"{search_url}&page={page}", headers=HEADERS).json()
            items = response.get('items', [])

            if not items:
                break

            for repo in items:
                name = repo['full_name']
                print(f"Checking {name}...", end=" ")

                is_valid, prs, ai_prs = filter_repository(name)

                if is_valid:
                    print(f"MATCH! (PRs: {prs}, AI: {ai_prs})")
                    discovered_repos.append({
                        "full_name": name,
                        "html_url": repo['html_url'],
                        "merged_prs_last_year": prs,
                        "ai_labeled_prs_last_year": ai_prs
                    })
                else:
                    print("Skipped.")

                if len(discovered_repos) >= target_count:
                    break
            page += 1
            time.sleep(2) # Avoid aggressive polling
        if len(discovered_repos) >= target_count:
            break

    # Save to data/raw/ as per project structure
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/repositories.json', 'w') as f:
        json.dump(discovered_repos, f, indent=4)

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"\nFinished! Acquired {len(discovered_repos)} repos in {duration} seconds.")

if __name__ == "__main__":
    # You can set target_count to 20 or 30 for your pilot study 
    main(target_count=100)