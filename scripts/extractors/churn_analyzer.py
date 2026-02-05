import os, time, json
from datetime import timedelta, datetime
from .github_utils import GitHubUtils
from github import GithubException

class ChurnAnalyzer:
    def __init__(self, cache_dir='data/cache/rework', retry_initial=5, retry_max=5):
        self.github = GitHubUtils()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.retry_initial = retry_initial
        self.retry_max = retry_max

    def _retry(self, fn, *args, **kwargs):
        wait = self.retry_initial
        for i in range(self.retry_max):
            try:
                return fn(*args, **kwargs)
            except GithubException as e:
                if e.status in (403, 429):
                    sleep = min(wait, 600)
                    print(f"  Rate-limited ({e.status}), sleeping {sleep}s")
                    time.sleep(sleep)
                    wait *= 2
                    continue
                raise
        raise RuntimeError("Retries exhausted")

    def compute_pr_rework(self, repo, pr, window_days=90, max_files=None, cache=True):
        key = f"{repo.full_name.replace('/', '__')}_pr{pr.number}.json"
        cache_path = os.path.join(self.cache_dir, key)
        if cache and os.path.exists(cache_path):
            return json.load(open(cache_path, 'r'))

        merged_at = pr.merged_at
        if not merged_at:
            return None

        since = merged_at
        until = merged_at + timedelta(days=window_days)

        files = [f.filename for f in pr.get_files()]
        if max_files:
            files = files[:max_files]

        per_file = []
        total_churn = 0
        total_events = 0
        first_dates = []
        last_dates = []

        for fpath in files:
            # get commits touching file in window (may be many API calls)
            commits = self._retry(repo.get_commits, path=fpath, since=since, until=until)
            churn_lines = 0
            events = 0
            first = None
            last = None

            for c in commits:
                # commit object contains list of files touched; get full commit to inspect files
                try:
                    full = self._retry(repo.get_commit, c.sha)
                except Exception as e:
                    print(f"   ✗ Failed to fetch commit {c.sha} for {repo.full_name}: {e}")
                    continue

                # inspect per-file stats
                for cf in full.files:
                    if cf.filename == fpath or cf.filename.endswith('/' + fpath.split('/')[-1]):
                        adds = cf.additions or 0
                        dels = cf.deletions or 0
                        churn_lines += adds + dels
                        events += 1
                        when = full.commit.author.date if full.commit and full.commit.author else None
                        if when:
                            if not first or when < first: first = when
                            if not last or when > last: last = when

            if events > 0:
                per_file.append({
                    'file': fpath,
                    'churn_lines': churn_lines,
                    'events': events,
                    'first': first.isoformat() if first else None,
                    'last': last.isoformat() if last else None
                })
                total_churn += churn_lines
                total_events += events
                if first: first_dates.append(first)
                if last: last_dates.append(last)

        result = {
            'repo_name': repo.full_name,
            'pr_number': pr.number,
            'merged_at': merged_at.isoformat(),
            'churn_lines_90d': total_churn,
            'rework_events': total_events,
            'files_reworked_count': len(per_file),
            'first_rework_date': min(first_dates).isoformat() if first_dates else None,
            'last_rework_date': max(last_dates).isoformat() if last_dates else None,
            'per_file_rework': per_file
        }

        if cache:
            with open(cache_path, 'w') as fh:
                json.dump(result, fh)

        return result