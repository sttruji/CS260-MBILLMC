"""Reusable GitHub API utilities."""
import os
from github import Github
from dotenv import load_dotenv

load_dotenv()

class GitHubUtils:
    """Helper class for GitHub API interactions."""
    
    def __init__(self):
        """Initialize GitHub client with token from environment."""
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")
        self.client = Github(token)
    
    def get_repository(self, repo_full_name):
        """
        Fetch a repository object by full name (owner/repo).
        
        Args:
            repo_full_name (str): Repository in format 'owner/repo'
        
        Returns:
            github.Repository.Repository: Repository object
        """
        try:
            return self.client.get_repo(repo_full_name)
        except Exception as e:
            print(f"Error fetching repository {repo_full_name}: {e}")
            return None
    
    def get_merged_prs(self, repo, max_prs=None):
        """
        Fetch all merged PRs from a repository.
        
        Args:
            repo: github.Repository.Repository object
            max_prs (int, optional): Maximum PRs to fetch (None = all)
        
        Yields:
            github.PullRequest.PullRequest: Merged PR object
        """
        try:
            prs = repo.get_pulls(state='closed', sort='updated', direction='desc')
            count = 0
            for pr in prs:
                if max_prs and count >= max_prs:
                    break
                if pr.merged:
                    yield pr
                    count += 1
        except Exception as e:
            print(f"Error fetching PRs from {repo.full_name}: {e}")
    
    def get_pr_labels(self, pr):
        """Extract PR labels as lowercase list."""
        return [label.name.lower() for label in pr.labels]
    
    def get_pr_metadata(self, pr):
        """
        Extract essential metadata from a PR.
        
        Returns:
            dict: PR metadata
        """
        return {
            'pr_number': pr.number,
            'title': pr.title,
            'body': pr.body or '',
            'labels': self.get_pr_labels(pr),
            'created_at': pr.created_at,
            'merged_at': pr.merged_at,
            'author': pr.user.login if pr.user else 'unknown',
            'additions': pr.additions,
            'deletions': pr.deletions,
            'changed_files': pr.changed_files,
            'commits': pr.commits
        }


    def get_pr_commit_stats(self, repo, pr, max_commits=None):
        """
        Yield commit-level stats for commits in a PR.
        Each yielded dict: {'sha', 'additions', 'deletions', 'total'}
        Note: Uses repo.get_commit(sha) to ensure .stats is populated (extra API calls).
        """
        count = 0
        try:
            for pr_commit in pr.get_commits():
                if max_commits and count >= max_commits:
                    break
                sha = pr_commit.sha
                try:
                    full_commit = repo.get_commit(sha)
                    stats = full_commit.stats
                    additions = stats.additions if stats and hasattr(stats, 'additions') else 0
                    deletions = stats.deletions if stats and hasattr(stats, 'deletions') else 0
                    total = (additions or 0) + (deletions or 0)
                except Exception:
                    # Best-effort fallback if get_commit fails
                    additions = 0
                    deletions = 0
                    total = 0
                yield {
                    'sha': sha,
                    'additions': additions or 0,
                    'deletions': deletions or 0,
                    'total': total or 0
                }
                count += 1
        except Exception as e:
            print(f"Error getting commits for PR {pr.number} in {repo.full_name}: {e}")
            return