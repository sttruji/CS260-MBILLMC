"""Compute per-PR volume metrics (NLOC, commits)"""
from .github_utils import GitHubUtils

class MetricsExtractor:
    def __init__(self):
        self.github_utils = GitHubUtils()

    def compute_pr_metrics(self, repo, pr, max_commits=None):
        """
        Returns:
          {
            'commit_count': int,
            'nloc_total': int,
            'avg_nloc_per_commit': float,
            'commit_stats': [ {'sha','additions','deletions','total'}, ... ]
          }
        """
        commit_stats = list(self.github_utils.get_pr_commit_stats(repo, pr, max_commits))
        commit_count = len(commit_stats)
        nloc_total = sum(c['total'] for c in commit_stats)
        avg_nloc = (nloc_total / commit_count) if commit_count else 0.0

        return {
            'commit_count': commit_count,
            'nloc_total': nloc_total,
            'avg_nloc_per_commit': round(avg_nloc, 2),
            'commit_stats': commit_stats
        }