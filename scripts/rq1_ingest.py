import argparse
import os
import sys
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging & paths
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLORED_REPOS_PATH = os.path.join(BASE_DIR, "data", "processed", "explored_repos.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "data", "processed", "rq1_checkpoint.parquet")
HUMAN_COMMITS_CACHE = os.path.join(BASE_DIR, "data", "processed", "human_pr_commits.parquet")
HUMAN_REVIEWS_CACHE = os.path.join(BASE_DIR, "data", "processed", "human_pr_reviews.parquet")
HUMAN_PR_STATS_CACHE = os.path.join(BASE_DIR, "data", "processed", "human_pr_stats.parquet")
HUMAN_PR_FILES_CACHE = os.path.join(BASE_DIR, "data", "processed", "human_pr_files.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "rq1_main_frame.parquet")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "results", "rq1_main_frame.csv")

load_dotenv(os.path.join(BASE_DIR, ".env"))


class RepoIngestor:
    def __init__(self, target_count):
        self.target_count = target_count
        self.explored_repos = {}  # repo_full_name -> {"created_at": ...}
        self.load_explored_repos()
        self.main_frame = None  # Main DataFrame with all PR data after transformations
        self.github_token = os.getenv("GITHUB_TOKEN")
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not set – GitHub API calls will be rate-limited.")

    def load_explored_repos(self):
        """Load previously explored repos from CSV into memory dict."""
        if os.path.exists(EXPLORED_REPOS_PATH):
            try:
                df = pd.read_csv(EXPLORED_REPOS_PATH)
                for _, row in df.iterrows():
                    self.explored_repos[row["repo_name"]] = {
                        "created_at": row.get("created_at"),
                        "timestamp": row.get("timestamp"),
                    }
                logger.info("Loaded %d explored repos from %s", len(self.explored_repos), EXPLORED_REPOS_PATH)
            except Exception as exc:
                logger.warning("Could not load explored repos: %s", exc)
        else:
            logger.info("No explored-repos file found – starting fresh.")

    def save_explored_repos(self):
        """Persist explored-repos registry back to CSV."""
        rows = [
            {"repo_name": name, "created_at": meta.get("created_at"), "timestamp": datetime.utcnow().isoformat()}
            for name, meta in self.explored_repos.items()
        ]
        os.makedirs(os.path.dirname(EXPLORED_REPOS_PATH), exist_ok=True)
        pd.DataFrame(rows).to_csv(EXPLORED_REPOS_PATH, index=False)
        logger.info("Saved %d explored repos.", len(rows))

    def already_explored(self, repo_name):
        '''
        Holds a set of already explored repo IDs to avoid redundant API calls and processing.
        Checks if repo_name is in memory dict of explored repos.
        '''
        return repo_name in self.explored_repos

    def _github_get(self, url):
        """Authenticated GitHub API GET with retry/backoff."""
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 403:
                    wait = max(int(resp.headers.get("X-RateLimit-Reset", 0)) - int(time.time()), 5)
                    logger.warning("Rate-limited. Sleeping %ds...", wait)
                    time.sleep(wait)
                    continue
                logger.warning("GitHub API %s returned %d", url, resp.status_code)
                return None
            except requests.RequestException as exc:
                logger.warning("GitHub request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------------ #
    #  FETCH human PR commit + review data via GitHub API
    # ------------------------------------------------------------------ #
    def fetch_human_pr_details(self, human_pr_df):
        """
        Fetch review data AND PR-level LOC stats for human PRs.

        Strategy (minimises API calls):
          • GET /repos/{o}/{r}/pulls/{n}          → additions, deletions  (1 call)
          • GET /repos/{o}/{r}/pulls/{n}/commits  → commit list for iteration/message data (1 call)
          • GET /repos/{o}/{r}/pulls/{n}/reviews  → review data (1 call)

        Three caches:
          HUMAN_PR_STATS_CACHE  – one row per PR: pr_id, pr_additions, pr_deletions
          HUMAN_COMMITS_CACHE   – one row per commit (sha, pr_id, author, message …)
          HUMAN_REVIEWS_CACHE   – one row per review

        Returns (commits_df, commit_details_df, reviews_df, pr_stats_df).
        """
        # ---- load existing caches ----
        cached_stats = pd.read_parquet(HUMAN_PR_STATS_CACHE) if os.path.exists(HUMAN_PR_STATS_CACHE) else pd.DataFrame()
        cached_commits = pd.read_parquet(HUMAN_COMMITS_CACHE) if os.path.exists(HUMAN_COMMITS_CACHE) else pd.DataFrame()
        cached_reviews = pd.read_parquet(HUMAN_REVIEWS_CACHE) if os.path.exists(HUMAN_REVIEWS_CACHE) else pd.DataFrame()

        cached_ids = set()
        if not cached_stats.empty:
            cached_ids = set(cached_stats["pr_id"].unique())
        # Also check commits cache for reviews-only PRs
        if not cached_commits.empty:
            cached_ids &= set(cached_commits["pr_id"].unique())  # require both
        else:
            cached_ids = set()  # need both caches present

        needed = human_pr_df[~human_pr_df["id"].isin(cached_ids)]
        if needed.empty and not cached_stats.empty:
            logger.info("Human PR caches complete (%d PRs). Skipping API.", len(cached_ids))
            return cached_commits, self._commits_to_details(cached_commits, cached_stats), cached_reviews, cached_stats

        logger.info("Fetching data for %d human PRs via GitHub API (cached: %d)...", len(needed), len(cached_ids))

        all_stats = []
        all_commits = []
        all_reviews = []
        total = len(needed)
        save_every = 200  # incremental save frequency

        for idx, (_, pr) in enumerate(needed.iterrows(), 1):
            pr_id = pr["id"]
            html_url = pr.get("html_url", "")
            try:
                parts = html_url.rstrip("/").split("/")
                owner, repo, pr_num = parts[-4], parts[-3], parts[-1]
            except (IndexError, ValueError):
                logger.warning("Cannot parse html_url: %s", html_url)
                continue

            # --- PR-level stats (additions/deletions) ---
            pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"
            pr_data = self._github_get(pr_url)
            if pr_data and isinstance(pr_data, dict):
                all_stats.append({
                    "pr_id": pr_id,
                    "pr_additions": pr_data.get("additions", 0),
                    "pr_deletions": pr_data.get("deletions", 0),
                    "pr_changed_files": pr_data.get("changed_files", 0),
                })
            else:
                all_stats.append({"pr_id": pr_id, "pr_additions": 0, "pr_deletions": 0, "pr_changed_files": 0})

            # --- commits (for messages / iteration count) ---
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/commits?per_page=100"
            commits_data = self._github_get(commits_url)
            if commits_data and isinstance(commits_data, list):
                for c in commits_data:
                    all_commits.append({
                        "sha": c.get("sha"),
                        "pr_id": pr_id,
                        "author": (c.get("author") or {}).get("login"),
                        "committer": (c.get("committer") or {}).get("login"),
                        "message": (c.get("commit") or {}).get("message"),
                    })

            # --- reviews ---
            reviews_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/reviews?per_page=100"
            reviews_data = self._github_get(reviews_url)
            if reviews_data and isinstance(reviews_data, list):
                for r in reviews_data:
                    all_reviews.append({
                        "id": r.get("id"),
                        "pr_id": pr_id,
                        "user": (r.get("user") or {}).get("login"),
                        "user_type": (r.get("user") or {}).get("type"),
                        "state": r.get("state"),
                        "submitted_at": r.get("submitted_at"),
                        "body": r.get("body"),
                    })

            if idx % 50 == 0 or idx == total:
                logger.info("  [%d/%d] human PRs fetched...", idx, total)

            # Incremental save to avoid losing progress
            if idx % save_every == 0:
                self._save_human_caches(cached_stats, cached_commits, cached_reviews,
                                        all_stats, all_commits, all_reviews)

        # ---- final merge with cache & save ----
        merged_stats, merged_commits, merged_reviews = self._save_human_caches(
            cached_stats, cached_commits, cached_reviews,
            all_stats, all_commits, all_reviews,
        )
        logger.info("Human PR caches saved: %d stats, %d commits, %d reviews.",
                     len(merged_stats), len(merged_commits), len(merged_reviews))

        return merged_commits, self._commits_to_details(merged_commits, merged_stats), merged_reviews, merged_stats

    @staticmethod
    def _save_human_caches(cached_stats, cached_commits, cached_reviews,
                           new_stats_list, new_commits_list, new_reviews_list):
        """Merge new data with cached data and persist all three caches."""
        os.makedirs(os.path.dirname(HUMAN_PR_STATS_CACHE), exist_ok=True)

        new_stats = pd.DataFrame(new_stats_list)
        merged_stats = pd.concat([cached_stats, new_stats], ignore_index=True)
        merged_stats.drop_duplicates(subset=["pr_id"], keep="last", inplace=True)
        if not merged_stats.empty:
            merged_stats.to_parquet(HUMAN_PR_STATS_CACHE, index=False)

        new_commits = pd.DataFrame(new_commits_list)
        merged_commits = pd.concat([cached_commits, new_commits], ignore_index=True)
        merged_commits.drop_duplicates(subset=["sha", "pr_id"], keep="last", inplace=True)
        if not merged_commits.empty:
            merged_commits.to_parquet(HUMAN_COMMITS_CACHE, index=False)

        new_reviews = pd.DataFrame(new_reviews_list)
        merged_reviews = pd.concat([cached_reviews, new_reviews], ignore_index=True)
        if not merged_reviews.empty:
            merged_reviews.to_parquet(HUMAN_REVIEWS_CACHE, index=False)

        return merged_stats, merged_commits, merged_reviews

    def fetch_human_pr_stats_only(self, human_pr_df):
        """
        Lightweight fetch: only GET /pulls/{n} for additions/deletions.
        1 API call per PR (vs 3 in full fetch).  Uses HUMAN_PR_STATS_CACHE.
        """
        cached_stats = pd.read_parquet(HUMAN_PR_STATS_CACHE) if os.path.exists(HUMAN_PR_STATS_CACHE) else pd.DataFrame()
        cached_ids = set(cached_stats["pr_id"].unique()) if not cached_stats.empty else set()

        needed = human_pr_df[~human_pr_df["id"].isin(cached_ids)]
        if needed.empty:
            logger.info("PR stats cache complete (%d PRs). Skipping API.", len(cached_ids))
            return cached_stats

        logger.info("Fetching PR-level stats for %d human PRs (cached: %d)...", len(needed), len(cached_ids))
        all_stats = []
        total = len(needed)

        for idx, (_, pr) in enumerate(needed.iterrows(), 1):
            pr_id = pr["id"]
            html_url = pr.get("html_url", "")
            try:
                parts = html_url.rstrip("/").split("/")
                owner, repo, pr_num = parts[-4], parts[-3], parts[-1]
            except (IndexError, ValueError):
                logger.warning("Cannot parse html_url: %s", html_url)
                continue

            pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"
            pr_data = self._github_get(pr_url)
            if pr_data and isinstance(pr_data, dict):
                all_stats.append({
                    "pr_id": pr_id,
                    "pr_additions": pr_data.get("additions", 0),
                    "pr_deletions": pr_data.get("deletions", 0),
                    "pr_changed_files": pr_data.get("changed_files", 0),
                })
            else:
                all_stats.append({"pr_id": pr_id, "pr_additions": 0, "pr_deletions": 0, "pr_changed_files": 0})

            if idx % 100 == 0 or idx == total:
                logger.info("  [%d/%d] PR stats fetched...", idx, total)

            # Incremental save every 500
            if idx % 500 == 0:
                new_df = pd.DataFrame(all_stats)
                merged = pd.concat([cached_stats, new_df], ignore_index=True)
                merged.drop_duplicates(subset=["pr_id"], keep="last", inplace=True)
                os.makedirs(os.path.dirname(HUMAN_PR_STATS_CACHE), exist_ok=True)
                merged.to_parquet(HUMAN_PR_STATS_CACHE, index=False)

        new_df = pd.DataFrame(all_stats)
        merged = pd.concat([cached_stats, new_df], ignore_index=True)
        merged.drop_duplicates(subset=["pr_id"], keep="last", inplace=True)
        os.makedirs(os.path.dirname(HUMAN_PR_STATS_CACHE), exist_ok=True)
        merged.to_parquet(HUMAN_PR_STATS_CACHE, index=False)
        logger.info("PR stats cache saved: %d rows.", len(merged))
        return merged

    def fetch_human_pr_files(self, human_pr_df):
        """
        Fetch the list of files modified by each human PR.
        Uses GET /repos/{o}/{r}/pulls/{n}/files (paginated, up to 300 files).
        Returns a DataFrame with columns: pr_id, filename
        Cached to HUMAN_PR_FILES_CACHE.
        """
        cached = pd.read_parquet(HUMAN_PR_FILES_CACHE) if os.path.exists(HUMAN_PR_FILES_CACHE) else pd.DataFrame()
        cached_ids = set(cached["pr_id"].unique()) if not cached.empty else set()

        needed = human_pr_df[~human_pr_df["id"].isin(cached_ids)]
        if needed.empty:
            logger.info("Human PR files cache complete (%d PRs). Skipping API.", len(cached_ids))
            return cached

        logger.info("Fetching file lists for %d human PRs (cached: %d)...", len(needed), len(cached_ids))
        all_rows = []
        total = len(needed)

        for idx, (_, pr) in enumerate(needed.iterrows(), 1):
            pr_id = pr["id"]
            html_url = pr.get("html_url", "")
            try:
                parts = html_url.rstrip("/").split("/")
                owner, repo, pr_num = parts[-4], parts[-3], parts[-1]
            except (IndexError, ValueError):
                continue

            files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files?per_page=100"
            files_data = self._github_get(files_url)
            if files_data and isinstance(files_data, list):
                for f in files_data:
                    all_rows.append({"pr_id": pr_id, "filename": f.get("filename")})
            # If no files returned, still mark this PR as fetched with a sentinel
            if not files_data or not isinstance(files_data, list) or len(files_data) == 0:
                all_rows.append({"pr_id": pr_id, "filename": None})

            if idx % 100 == 0 or idx == total:
                logger.info("  [%d/%d] human PR file lists fetched...", idx, total)

            # Incremental save every 500
            if idx % 500 == 0:
                new_df = pd.DataFrame(all_rows)
                merged = pd.concat([cached, new_df], ignore_index=True)
                merged.drop_duplicates(subset=["pr_id", "filename"], keep="last", inplace=True)
                os.makedirs(os.path.dirname(HUMAN_PR_FILES_CACHE), exist_ok=True)
                merged.to_parquet(HUMAN_PR_FILES_CACHE, index=False)

        new_df = pd.DataFrame(all_rows)
        merged = pd.concat([cached, new_df], ignore_index=True)
        merged.drop_duplicates(subset=["pr_id", "filename"], keep="last", inplace=True)
        os.makedirs(os.path.dirname(HUMAN_PR_FILES_CACHE), exist_ok=True)
        merged.to_parquet(HUMAN_PR_FILES_CACHE, index=False)
        logger.info("Human PR files cache saved: %d rows across %d PRs.",
                     len(merged), merged["pr_id"].nunique())
        return merged

    @staticmethod
    def _commits_to_details(commits_df, pr_stats_df=None):
        """
        Build a commit_details-like DataFrame from commits + PR-level stats.
        We distribute PR-level additions/deletions evenly across commits for
        compatibility with the AIDev pr_commit_details schema.
        If pr_stats_df is None, stats columns default to 0.
        """
        if commits_df.empty:
            return pd.DataFrame(columns=[
                "sha", "pr_id", "author", "committer", "message",
                "commit_stats_total", "commit_stats_additions", "commit_stats_deletions",
                "filename", "status", "additions", "deletions", "changes", "patch",
            ])
        details = commits_df.copy()

        if pr_stats_df is not None and not pr_stats_df.empty:
            # Count commits per PR for even distribution
            commit_counts = details.groupby("pr_id").size().reset_index(name="_n_commits")
            details = details.merge(commit_counts, on="pr_id", how="left")
            details = details.merge(
                pr_stats_df[["pr_id", "pr_additions", "pr_deletions"]],
                on="pr_id", how="left",
            )
            details["pr_additions"] = details["pr_additions"].fillna(0)
            details["pr_deletions"] = details["pr_deletions"].fillna(0)
            details["_n_commits"] = details["_n_commits"].clip(lower=1)
            details["commit_stats_additions"] = (details["pr_additions"] / details["_n_commits"]).round().astype(int)
            details["commit_stats_deletions"] = (details["pr_deletions"] / details["_n_commits"]).round().astype(int)
            details["commit_stats_total"] = details["commit_stats_additions"] + details["commit_stats_deletions"]
            details.drop(columns=["_n_commits", "pr_additions", "pr_deletions"], inplace=True, errors="ignore")
        else:
            details["commit_stats_additions"] = 0
            details["commit_stats_deletions"] = 0
            details["commit_stats_total"] = 0

        details["filename"] = None
        details["status"] = None
        details["additions"] = details["commit_stats_additions"]
        details["deletions"] = details["commit_stats_deletions"]
        details["changes"] = details["commit_stats_total"]
        details["patch"] = None
        return details

    # ------------------------------------------------------------------ #
    #  CHECKPOINT / SAVE helpers
    # ------------------------------------------------------------------ #
    def save_checkpoint(self):
        """Save main_frame to parquet checkpoint."""
        if self.main_frame is not None and not self.main_frame.empty:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            self.main_frame.to_parquet(CHECKPOINT_PATH, index=False)
            logger.info("Checkpoint saved: %d rows → %s", len(self.main_frame), CHECKPOINT_PATH)

    def load_checkpoint(self):
        """Load checkpoint if exists. Returns True on success."""
        if os.path.exists(CHECKPOINT_PATH):
            try:
                self.main_frame = pd.read_parquet(CHECKPOINT_PATH)
                logger.info("Checkpoint loaded: %d rows from %s", len(self.main_frame), CHECKPOINT_PATH)
                return True
            except Exception as exc:
                logger.warning("Failed to load checkpoint: %s", exc)
        return False

    def save_final(self):
        """Save final main_frame to Parquet + CSV."""
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("Nothing to save – main_frame is empty.")
            return
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        self.main_frame.to_parquet(OUTPUT_PATH, index=False)
        self.main_frame.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info("Final output saved: %d rows → %s / %s", len(self.main_frame), OUTPUT_PATH, OUTPUT_CSV_PATH)


    def check_stars(self, all_repo_df, all_pr_df):
        '''
        Docstring for check_stars
        
        :param all_repo_df: All repos in AIDev dataset with metadata 
        :param all_pr_df: All prs in AIdev dataset with metadata 
        For each repo check the number of stars are:
            < 500: drop all associated PRs via all_pr_df['id'].isin(repo_pr_ids)
            = 500-1k: Add stars to pior and new column for star_group (0) via all_pr_df['id'].isin(repo_pr_ids) 
            = 1k-5k: Add stars to pior and new column for star_group (1) via all_pr_df['id'].isin(repo_pr_ids)
            > 5k: Add stars to pior and new column for star_group (2) via all_pr_df['id'].isin(repo_pr_ids) 

        transformed all_pr_df saved to RepoIngestor main_frame with new columns for stars and star_group, and filtered to only include PRs from repos with >= 500 stars, also merge with all_repo_df to include repo metadata for future transformations
        '''
        logger.info("check_stars: filtering repos by star count...")

        qualified = all_repo_df[all_repo_df["stars"] >= 500].copy()
        logger.info("Repos with >=500 stars: %d / %d", len(qualified), len(all_repo_df))

        def _star_group(s):
            if s < 1000: return 0
            if s < 5000: return 1
            return 2

        qualified["star_group"] = qualified["stars"].apply(_star_group)

        # Rename repo cols to avoid collision
        qualified = qualified.rename(columns={
            "id": "repo_id", "url": "repo_api_url", "full_name": "repo_full_name",
            "language": "repo_language", "license": "repo_license",
            "forks": "repo_forks", "stars": "repo_stars",
        })

        merged = all_pr_df.merge(qualified, on="repo_id", how="inner")
        logger.info("PRs after star filter: %d", len(merged))
        self.main_frame = merged


    def check_age(self):
        '''
        Docstring for check_age

        With main_frame containing all PRs from repos with >= 500 stars, check the age of each repo at the time of each PR creation and bucket into age groups.
        For each PR check Repos age at the time of PR creation (pr "created_at" - Repos creation date)

        To determine repos creation date:
        1. use already_explored() to check if repo has already been explored and if so, look up creation date from in-memory set of explored repos load_explored_repos() at initialization
        2. If not already explored, make API call to GitHub to get repo creation date /
            with repo_url and .env GITHUB_TOKEN for authentication, then save to in-memory set of explored repos for future lookups

        Add column for repos age at time of pr in these buckets:
            < 1 year: age_group (0)
            1 - 2 years: age_group (1)
            2 - 5 years: age_group (2)
            > 5 years: age_group (3)

        transformed all_pr_df saved to RepoIngestor main_frame with new column for age_group
        '''
        logger.info("check_age: computing repo ages...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping check_age"); return

        self.main_frame["created_at"] = pd.to_datetime(self.main_frame["created_at"], errors="coerce", utc=True)
        unique_repos = self.main_frame["repo_full_name"].dropna().unique()
        logger.info("Unique repos to look up creation date: %d", len(unique_repos))

        creation_map = {}
        for repo_name in unique_repos:
            if self.already_explored(repo_name) and self.explored_repos[repo_name].get("created_at"):
                creation_map[repo_name] = pd.to_datetime(self.explored_repos[repo_name]["created_at"], utc=True)
            else:
                api_url = f"https://api.github.com/repos/{repo_name}"
                data = self._github_get(api_url)
                if data and "created_at" in data:
                    creation_map[repo_name] = pd.to_datetime(data["created_at"], utc=True)
                    self.explored_repos[repo_name] = {"created_at": data["created_at"]}
                else:
                    creation_map[repo_name] = None
                    logger.warning("Could not get creation date for %s", repo_name)

        self.save_explored_repos()

        self.main_frame["repo_created_at"] = self.main_frame["repo_full_name"].map(creation_map)
        self.main_frame["repo_created_at"] = pd.to_datetime(self.main_frame["repo_created_at"], errors="coerce", utc=True)

        age_years = (self.main_frame["created_at"] - self.main_frame["repo_created_at"]).dt.total_seconds() / (365.25 * 24 * 3600)

        def _age_group(y):
            if pd.isna(y): return None
            if y < 1: return 0
            if y < 2: return 1
            if y < 5: return 2
            return 3

        self.main_frame["repo_age_years"] = age_years
        self.main_frame["age_group"] = age_years.apply(_age_group)
        logger.info("check_age: done. Non-null age_group: %d", self.main_frame["age_group"].notna().sum())


    def check_contributors(self, all_user_df):
        '''
        Docstring for check_contributors
        
        :param all_user_df: All users in AIDev dataset with metadata

        For each unique "user" in all_user_df increment count of contributiors in /
        corresponding repo in main_frame via all_pr_df['id'].isin(repo_pr_ids) and /
        add new column for contributor_count

        additionally add column for contributor_count buckets:
            small team: < 5 contributors: contributor_group (0)
            medium team: 5 - 20 contributors: contributor_group (1)
            large team: > 20 contributors: contributor_group (2)

        transformed all_pr_df saved to RepoIngestor main_frame with new column for contributor_count
        '''
        logger.info("check_contributors: counting per-repo contributors...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        contrib_counts = (
            self.main_frame.groupby("repo_full_name")["user"]
            .nunique()
            .reset_index()
            .rename(columns={"user": "contributor_count"})
        )
        self.main_frame = self.main_frame.merge(contrib_counts, on="repo_full_name", how="left")

        def _contrib_group(n):
            if pd.isna(n): return None
            if n < 5: return 0
            if n <= 20: return 1
            return 2

        self.main_frame["contributor_group"] = self.main_frame["contributor_count"].apply(_contrib_group)
        logger.info("check_contributors: done.")


    def check_ai_prs(self, all_user_df):
        '''
        Docstring for check_ai_prs
        
        :param self: RepoIngestor instance with main_frame 

        For each PR in main_frame, check "agent" field 
        If agent = "Human", add 0 to new column ai_pr
        If agent = <Any other input> ("Claude_Code"), add 1 to new column ai_pr

        merge main_frame with all_user_df to include user metadata for future transformations
        '''
        logger.info("check_ai_prs: classifying AI vs human PRs...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        self.main_frame["ai_pr"] = self.main_frame["agent"].fillna("Human").apply(lambda x: 0 if x == "Human" else 1)

        user_meta = all_user_df.rename(columns={
            "id": "user_meta_id", "login": "user_login",
            "followers": "user_followers", "following": "user_following",
            "created_at": "user_created_at",
        })
        self.main_frame = self.main_frame.merge(
            user_meta[["user_login", "user_followers", "user_following", "user_created_at"]],
            left_on="user", right_on="user_login", how="left",
        )
        self.main_frame.drop(columns=["user_login"], inplace=True, errors="ignore")
        logger.info("check_ai_prs: done. AI=%d, Human=%d",
                     (self.main_frame["ai_pr"] == 1).sum(),
                     (self.main_frame["ai_pr"] == 0).sum())


    def check_task_type(self, pr_task_type_df):
        '''
        Docstring for check_task_type

        :param pr_task_type_df: DataFrame with PR task type classifications
        For each PR in main_frame, check pr_task_type_df for corresponding PR id and add new column for task type (feat, fix, docs, refactor, test)

        Look up pr id in mainframe and add new column for task type based on pr_task_type_df classification for each PR id
        pr_task_type_df["type"] and pr_task_type_df['id'] to load main frame pr with new column for task type

        add new column for task type buckets:
            feature: feat: task_type_group (0)
            bug fix: fix: task_type_group (1)
            documentation: docs: task_type_group (2)
            refactor: refactor: task_type_group (3)
            test: test: task_type_group (4)

        transformed main_frame saved to RepoIngestor main_frame with new column for task type and task_type_group
        '''
        logger.info("check_task_type: mapping task types...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        task_map = pr_task_type_df[["id", "type"]].rename(columns={"type": "task_type"})
        self.main_frame = self.main_frame.merge(task_map, on="id", how="left")

        type_to_group = {"feat": 0, "fix": 1, "docs": 2, "refactor": 3, "test": 4}
        self.main_frame["task_type_group"] = (
            self.main_frame["task_type"].str.lower().str.strip().map(type_to_group)
        )
        logger.info("check_task_type: done.")


    def check_domain_type(self):
        '''
        Docstring for check_domain_type

        Hassatr for language field
        For each pr in main_frame check what "language" the pr is associated with 
        add a new column for domain type buckets:
            web development: JavaScript, HTML, CSS: domain_type_group (0)
            data science: Python, R: domain_type_group (1)
            mobile development: Java, Kotlin, Swift: domain_type_group (2)
            systems programming: C, C++: domain_type_group (3)
            other: all other languages: domain_type_group (4)

        transformed main_frame saved to RepoIngestor main_frame with new column for domain type and domain_type_group
        '''
        logger.info("check_domain_type: assigning domain types...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        domain_map = {
            "JavaScript": 0, "HTML": 0, "CSS": 0, "TypeScript": 0,
            "Python": 1, "R": 1, "Jupyter Notebook": 1,
            "Java": 2, "Kotlin": 2, "Swift": 2,
            "C": 3, "C++": 3,
        }
        self.main_frame["domain_type_group"] = (
            self.main_frame["repo_language"].map(domain_map).fillna(4).astype(int)
        )
        logger.info("check_domain_type: done.")


    def language_type(self):
        '''
        Docstring for language_type

        Hassatr for language field
        For each pr in main_frame check what "language" the pr is associated with 
        add a new column for language type buckets:
            statically typed: Java, C, C++, Go, Rust, TypeScript: language_type_group (0)
            dynamically typed: Python, JavaScript, Ruby, PHP: language_type_group (1)
            other: all other languages: language_type_group (2)

        transformed main_frame saved to RepoIngestor main_frame with new column for language type and language_group
        '''
        logger.info("language_type: classifying language typing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        static = {"Java", "C", "C++", "Go", "Rust", "TypeScript", "Kotlin", "Swift", "C#"}
        dynamic = {"Python", "JavaScript", "Ruby", "PHP", "Lua", "Perl", "R"}

        def _lang_group(lang):
            if pd.isna(lang): return 2
            if lang in static: return 0
            if lang in dynamic: return 1
            return 2

        self.main_frame["language_type_group"] = self.main_frame["repo_language"].apply(_lang_group)
        logger.info("language_type: done.")

    
    def time_to_first_review(self, pr_reviews_df):
        '''
        Docstring for time_to_first_review

        :param pr_reviews_df: DataFrame with PR review data including timestamps
        For each PR in main_frame, check pr_reviews_df for corresponding PR id and calculate time to first review (first review "submitted_at" - PR "created_at") and add new column for time_to_first_review

        transformed main_frame saved to RepoIngestor main_frame with new column for time to first review
        '''
        logger.info("time_to_first_review: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        pr_reviews_df = pr_reviews_df.copy()
        pr_reviews_df["submitted_at"] = pd.to_datetime(pr_reviews_df["submitted_at"], errors="coerce", utc=True)

        first_review = (
            pr_reviews_df.sort_values("submitted_at")
            .groupby("pr_id")["submitted_at"].first()
            .reset_index()
            .rename(columns={"submitted_at": "first_review_at"})
        )
        self.main_frame = self.main_frame.merge(first_review, left_on="id", right_on="pr_id", how="left")
        self.main_frame.drop(columns=["pr_id"], inplace=True, errors="ignore")

        self.main_frame["created_at"] = pd.to_datetime(self.main_frame["created_at"], errors="coerce", utc=True)
        delta = self.main_frame["first_review_at"] - self.main_frame["created_at"]
        self.main_frame["time_to_first_review"] = delta.dt.total_seconds() / 3600  # hours
        logger.info("time_to_first_review: done. Non-null: %d", self.main_frame["time_to_first_review"].notna().sum())


    def time_to_resolution(self):
        '''
        Docstring for time_to_resolution
        
        :param self: Description

        For each PR in main_frame:
            if PR "merged_at" is not null, calculate time to resolution (PR "merged_at" - PR "created_at") and add new column for time_to_resolution
            if PR "merged_at" is null, add null to time_to_resolution (PR "closed_at" - PR "created_at") and add new column for time_to_resolution
            additionally add rejected or accepted based on merged_at field (if merged_at is not null, accepted, if merged_at is null and closed_at is not null, rejected) to new column for pr_outcome

        transformed main_frame saved to RepoIngestor main_frame with new column for time to resolution and pr_outcome
        '''
        logger.info("time_to_resolution: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        self.main_frame["created_at"] = pd.to_datetime(self.main_frame["created_at"], errors="coerce", utc=True)
        self.main_frame["merged_at"] = pd.to_datetime(self.main_frame["merged_at"], errors="coerce", utc=True)
        self.main_frame["closed_at"] = pd.to_datetime(self.main_frame["closed_at"], errors="coerce", utc=True)

        resolution_dt = self.main_frame["merged_at"].fillna(self.main_frame["closed_at"])
        self.main_frame["time_to_resolution"] = (resolution_dt - self.main_frame["created_at"]).dt.total_seconds() / 3600  # hours

        def _outcome(row):
            if pd.notna(row["merged_at"]): return "accepted"
            if pd.notna(row["closed_at"]): return "rejected"
            return "open"

        self.main_frame["pr_outcome"] = self.main_frame.apply(_outcome, axis=1)
        logger.info("time_to_resolution: done.")

    
    def pr_size_LOC(self, pr_commit_details_df, pr_stats_df=None):
        '''
        Compute pr_size_loc, pr_additions, pr_deletions.

        For human PRs the PR-level stats come from pr_stats_df (fetched via
        the GitHub single-PR endpoint which is authoritative).  For AI PRs
        the stats are summed from pr_commit_details_df as before.

        If pr_stats_df is provided, human PR stats are taken directly from it
        rather than from commit details (which may be approximate).
        '''
        logger.info("pr_size_LOC: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        # ---- drop old columns if recomputing ----
        for col in ["pr_size_loc", "pr_additions", "pr_deletions"]:
            if col in self.main_frame.columns:
                self.main_frame.drop(columns=[col], inplace=True)

        # ---- AI PRs: aggregate from commit details ----
        commit_level = (
            pr_commit_details_df
            .drop_duplicates(subset=["sha", "pr_id"])
            .groupby("pr_id")
            .agg(pr_additions=("commit_stats_additions", "sum"),
                 pr_deletions=("commit_stats_deletions", "sum"))
            .reset_index()
        )
        commit_level["pr_size_loc"] = commit_level["pr_additions"] + commit_level["pr_deletions"]

        self.main_frame = self.main_frame.merge(
            commit_level[["pr_id", "pr_size_loc", "pr_additions", "pr_deletions"]],
            left_on="id", right_on="pr_id", how="left",
        )
        self.main_frame.drop(columns=["pr_id"], inplace=True, errors="ignore")

        # ---- Human PRs: overwrite with authoritative PR-level stats ----
        if pr_stats_df is not None and not pr_stats_df.empty:
            human_mask = self.main_frame["ai_pr"] == 0 if "ai_pr" in self.main_frame.columns else pd.Series(False, index=self.main_frame.index)
            if human_mask.any():
                stats_map = pr_stats_df.set_index("pr_id")
                human_ids = self.main_frame.loc[human_mask, "id"]
                matched_add = human_ids.map(stats_map["pr_additions"]).fillna(0).astype(int)
                matched_del = human_ids.map(stats_map["pr_deletions"]).fillna(0).astype(int)
                self.main_frame.loc[human_mask, "pr_additions"] = matched_add.values
                self.main_frame.loc[human_mask, "pr_deletions"] = matched_del.values
                self.main_frame.loc[human_mask, "pr_size_loc"] = (matched_add + matched_del).values
                logger.info("  Overwrote %d human PRs with authoritative PR-level stats.", human_mask.sum())

        logger.info("pr_size_LOC: done. Non-zero LOC: %d / %d",
                     (self.main_frame["pr_size_loc"].fillna(0) > 0).sum(), len(self.main_frame))


    def defect_density(self, pr_files_df=None, related_issue_df=None,
                       issue_df=None, pr_commits_df=None):
        """
        File-overlap defect detection strategy
        =======================================
        For every merged PR, open a 90-day window [merged_at, merged_at+90d].
        A follow-up "fix" PR is counted as a defect if **it touches at least
        one file that the original PR also modified**.

        Primary signal (file overlap):
          1. Build pr_id → set(filenames) from pr_files_df.
          2. For each merged original PR, find all fix PRs in same repo
             within the 90-day window.
          3. If the fix PR's file set overlaps with the original PR's file
             set → count as defect.

        Secondary signals (fallback when file data is missing):
          S2 – PR-number reference in fix PR title/body.
          S3 – Shared issue linkage via related_issue table.
          S4 – Revert detection (title contains "revert" + PR# or title match).

        Columns added / overwritten:
          defect_count_90d, has_defect_90d, defect_density
        """
        import re as _re

        logger.info("defect_density: computing 90-day post-merge defects (file-overlap strategy)...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        # ---- drop old columns if re-computing ----
        for col in ["defect_count_90d", "has_defect_90d", "defect_density"]:
            if col in self.main_frame.columns:
                self.main_frame.drop(columns=[col], inplace=True)

        self.main_frame["merged_at"] = pd.to_datetime(self.main_frame["merged_at"], errors="coerce", utc=True)
        self.main_frame["created_at"] = pd.to_datetime(self.main_frame["created_at"], errors="coerce", utc=True)

        # ---- build pr_id → set(filenames) index ----
        pr_to_files = {}  # {pr_id: set(filename, ...)}
        if pr_files_df is not None and not pr_files_df.empty:
            for pr_id, grp in pr_files_df.groupby("pr_id"):
                fnames = set(grp["filename"].dropna())
                if fnames:
                    pr_to_files[int(pr_id)] = fnames
            logger.info("File index built: %d PRs have file data.", len(pr_to_files))
        else:
            logger.warning("No pr_files_df provided – file-overlap signal disabled.")

        # ---- identify fix PRs (task_type == fix) ----
        is_fix = self.main_frame["task_type"].fillna("").str.lower().str.strip() == "fix"
        fix_prs = self.main_frame.loc[
            is_fix, ["id", "number", "repo_full_name", "created_at", "title", "body"]
        ].copy()
        fix_prs["created_at"] = pd.to_datetime(fix_prs["created_at"], errors="coerce", utc=True)

        # ---- secondary signal setup ----
        # S2: PR-number references from fix PR title+body
        def _extract_refs(title, body):
            text = str(title or "") + " " + str(body or "")
            return {int(m) for m in _re.findall(r'#(\d+)', text)}

        fix_prs["refs"] = fix_prs.apply(lambda r: _extract_refs(r["title"], r["body"]), axis=1)
        fix_prs["title_lower"] = fix_prs["title"].fillna("").str.lower()

        # S3: issue linkage maps
        pr_to_issues = {}   # {pr_id: {issue_id, ...}}
        issue_to_prs = {}   # {issue_id: {pr_id, ...}}
        if related_issue_df is not None and not related_issue_df.empty:
            for _, ri_row in related_issue_df.iterrows():
                pid, iid = ri_row["pr_id"], ri_row["issue_id"]
                if pd.notna(pid) and pd.notna(iid):
                    pid, iid = int(pid), int(iid)
                    pr_to_issues.setdefault(pid, set()).add(iid)
                    issue_to_prs.setdefault(iid, set()).add(pid)
            logger.info("Issue linkage built: %d PRs → issues, %d issues → PRs",
                         len(pr_to_issues), len(issue_to_prs))

        # ---- group fix PRs by repo ----
        fix_by_repo = dict(list(fix_prs.groupby("repo_full_name")))

        # ---- scan every merged PR ----
        merged_mask = self.main_frame["merged_at"].notna()
        merged_prs = self.main_frame.loc[
            merged_mask, ["id", "number", "repo_full_name", "merged_at", "title"]
        ].copy()
        logger.info("Scanning %d merged PRs for 90-day defect signals...", len(merged_prs))

        defect_counts = []
        signal_stats = {"file_overlap": 0, "pr_ref": 0, "issue_link": 0, "revert": 0}

        for _, row in merged_prs.iterrows():
            pr_id = row["id"]
            pr_number = row["number"]
            repo = row["repo_full_name"]
            merge_dt = row["merged_at"]
            window_end = merge_dt + timedelta(days=90)
            orig_title_lower = str(row.get("title") or "").lower()
            orig_files = pr_to_files.get(pr_id, set())
            count = 0

            if repo not in fix_by_repo:
                defect_counts.append({"id": pr_id, "defect_count_90d": 0})
                continue

            candidates = fix_by_repo[repo]
            in_window = candidates[
                (candidates["created_at"] > merge_dt) &
                (candidates["created_at"] <= window_end) &
                (candidates["id"] != pr_id)
            ]

            if in_window.empty:
                defect_counts.append({"id": pr_id, "defect_count_90d": 0})
                continue

            for _, fix_row in in_window.iterrows():
                fix_id = fix_row["id"]
                matched = False

                # PRIMARY: file-overlap – fix PR touches ≥1 of same files
                if not matched and orig_files:
                    fix_files = pr_to_files.get(fix_id, set())
                    if fix_files and orig_files & fix_files:
                        count += 1; signal_stats["file_overlap"] += 1; matched = True

                # SECONDARY S2: fix PR title/body references original PR#
                if not matched and pd.notna(pr_number):
                    fix_refs = fix_row["refs"]
                    if int(pr_number) in fix_refs:
                        count += 1; signal_stats["pr_ref"] += 1; matched = True

                # SECONDARY S3: shared issue linkage
                if not matched:
                    orig_issues = pr_to_issues.get(pr_id, set())
                    fix_issues = pr_to_issues.get(fix_id, set())
                    if orig_issues & fix_issues:
                        count += 1; signal_stats["issue_link"] += 1; matched = True

                # SECONDARY S4: revert detection
                if not matched:
                    fix_title_lower = fix_row["title_lower"]
                    if "revert" in fix_title_lower:
                        if pd.notna(pr_number) and str(int(pr_number)) in fix_title_lower:
                            count += 1; signal_stats["revert"] += 1; matched = True
                        elif orig_title_lower and len(orig_title_lower) > 10:
                            title_words = [w for w in orig_title_lower.split() if len(w) > 3]
                            if title_words and sum(1 for w in title_words if w in fix_title_lower) >= len(title_words) * 0.5:
                                count += 1; signal_stats["revert"] += 1; matched = True

            defect_counts.append({"id": pr_id, "defect_count_90d": count})

        defect_df = pd.DataFrame(defect_counts)
        self.main_frame = self.main_frame.merge(defect_df, on="id", how="left")
        self.main_frame["defect_count_90d"] = self.main_frame["defect_count_90d"].fillna(0).astype(int)
        self.main_frame["has_defect_90d"] = (self.main_frame["defect_count_90d"] > 0).astype(int)

        if "pr_size_loc" in self.main_frame.columns:
            self.main_frame["defect_density"] = self.main_frame.apply(
                lambda r: r["defect_count_90d"] / r["pr_size_loc"]
                if pd.notna(r["pr_size_loc"]) and r["pr_size_loc"] > 0 else None, axis=1)
        else:
            self.main_frame["defect_density"] = None

        total_defects = int(self.main_frame["has_defect_90d"].sum())
        logger.info("defect_density: done. PRs with ≥1 linked defect: %d / %d", total_defects, len(merged_prs))
        logger.info("Signal breakdown: file_overlap=%d  pr_ref=%d  issue_link=%d  revert=%d",
                     signal_stats["file_overlap"], signal_stats["pr_ref"],
                     signal_stats["issue_link"], signal_stats["revert"])


    def fix_resolution_time(self):
        '''
        Docstring for fix_resolution_time

        :param self: Description

        task type -> fix 

        merged_at - created_at for PRs with task type fix to calculate fix resolution time and add new column for fix_resolution_time

        transformed main_frame saved to RepoIngestor main_frame with new column for fix resolution time
        '''
        logger.info("fix_resolution_time: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        self.main_frame["fix_resolution_time"] = None
        is_fix = self.main_frame["task_type"].fillna("").str.lower().str.strip() == "fix"
        merged = self.main_frame["merged_at"].notna()
        mask = is_fix & merged
        delta = (self.main_frame.loc[mask, "merged_at"] - self.main_frame.loc[mask, "created_at"]).dt.total_seconds() / 3600
        self.main_frame.loc[mask, "fix_resolution_time"] = delta
        logger.info("fix_resolution_time: done.")


    def fix_size(self, pr_commit_details_df):
        '''
        Docstring for fix_size

        :param self: Description

        task type -> fix 

        Map main_fram['id'] to pr_commit_details_df['pr_id'] to get PR size in lines of code for PRs with task type fix
        skip if we already have a fix_size column in main_frame to avoid redundant calculations
        commit_stats_additions and commit_stats_deletions to calculate total lines of code changed for each PR with task type fix and add new column for fix_size
        '''
        logger.info("fix_size: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return
        if "fix_size" in self.main_frame.columns:
            logger.info("fix_size already exists – skipping."); return

        is_fix = self.main_frame["task_type"].fillna("").str.lower().str.strip() == "fix"
        self.main_frame["fix_size"] = None

        if "pr_size_loc" in self.main_frame.columns:
            self.main_frame.loc[is_fix, "fix_size"] = self.main_frame.loc[is_fix, "pr_size_loc"]
        else:
            fix_ids = set(self.main_frame.loc[is_fix, "id"])
            subset = pr_commit_details_df[pr_commit_details_df["pr_id"].isin(fix_ids)]
            loc = (
                subset.drop_duplicates(subset=["sha", "pr_id"])
                .groupby("pr_id")
                .agg(fix_add=("commit_stats_additions", "sum"), fix_del=("commit_stats_deletions", "sum"))
                .reset_index()
            )
            loc["fix_size"] = loc["fix_add"] + loc["fix_del"]
            self.main_frame = self.main_frame.merge(loc[["pr_id", "fix_size"]], left_on="id", right_on="pr_id", how="left", suffixes=("", "_calc"))
            if "fix_size_calc" in self.main_frame.columns:
                self.main_frame["fix_size"] = self.main_frame["fix_size_calc"].combine_first(self.main_frame["fix_size"])
                self.main_frame.drop(columns=["fix_size_calc"], inplace=True, errors="ignore")
            self.main_frame.drop(columns=["pr_id"], inplace=True, errors="ignore")
        logger.info("fix_size: done.")


    def fix_iteration_count(self, pr_commits_df):
        '''
        Docstring for fix_iteration_count

        :param self: Description

        task type -> fix 

        For each PR with task type fix
        go into pr_commits and for evey unique sha associated with the fixed pr accepted create new column with iteration count for accepeted fix prs 
        '''
        logger.info("fix_iteration_count: computing...")
        if self.main_frame is None or self.main_frame.empty:
            logger.warning("main_frame empty – skipping"); return

        is_fix = self.main_frame["task_type"].fillna("").str.lower().str.strip() == "fix"
        is_merged = self.main_frame["merged_at"].notna()
        fix_merged_ids = set(self.main_frame.loc[is_fix & is_merged, "id"])

        if not fix_merged_ids:
            self.main_frame["fix_iteration_count"] = None
            logger.info("No merged fix PRs found."); return

        subset = pr_commits_df[pr_commits_df["pr_id"].isin(fix_merged_ids)]
        counts = (
            subset.groupby("pr_id")["sha"].nunique()
            .reset_index()
            .rename(columns={"sha": "fix_iteration_count"})
        )
        self.main_frame = self.main_frame.merge(counts, left_on="id", right_on="pr_id", how="left")
        self.main_frame.drop(columns=["pr_id"], inplace=True, errors="ignore")
        logger.info("fix_iteration_count: done.")


def main():
    parser = argparse.ArgumentParser(description="Ingest GitHub PR data for RQ1 analysis")
    parser.add_argument('--target_count', type=int, default=None, help='Max number of unique repos to process (default: all)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--skip-age', action='store_true', help='Skip check_age (requires GitHub API calls)')
    parser.add_argument('--recompute-defects', action='store_true',
                        help='Load checkpoint, recompute only defect_density columns, save, and exit.')
    parser.add_argument('--recompute-loc', action='store_true',
                        help='Load checkpoint, refetch human PR stats, recompute pr_size_loc, save, and exit.')
    args = parser.parse_args()

    ingestor = RepoIngestor(target_count=args.target_count)

    # ------------------------------------------------------------------
    # Resume from checkpoint if requested
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Fast path: recompute only defect columns on existing checkpoint
    # ------------------------------------------------------------------
    if args.recompute_defects:
        if not ingestor.load_checkpoint():
            logger.error("No checkpoint found – run full pipeline first.")
            sys.exit(1)
        logger.info("Recomputing defect_density on existing checkpoint...")

        # ---- Load file data for file-overlap signal ----
        # AI PR files from pr_commit_details
        pr_commit_details_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
        ai_files = pr_commit_details_df[["pr_id", "filename"]].dropna(subset=["filename"]).drop_duplicates()

        # Human PR files from cache (fetch if missing)
        human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
        human_files = ingestor.fetch_human_pr_files(human_pr_df)

        # Combine into one pr_files_df
        pr_files_df = pd.concat([ai_files, human_files], ignore_index=True)
        pr_files_df = pr_files_df.dropna(subset=["filename"]).drop_duplicates(subset=["pr_id", "filename"])
        logger.info("Combined file data: %d file rows across %d PRs.",
                     len(pr_files_df), pr_files_df["pr_id"].nunique())

        # ---- Load secondary signal data ----
        related_issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/related_issue.parquet")
        issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/issue.parquet")

        ingestor.defect_density(pr_files_df, related_issue_df, issue_df)
        ingestor.save_checkpoint()
        ingestor.save_final()
        logger.info("Defect columns recomputed and saved. Done.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Fast path: recompute only pr_size_loc on existing checkpoint
    # ------------------------------------------------------------------
    if args.recompute_loc:
        if not ingestor.load_checkpoint():
            logger.error("No checkpoint found – run full pipeline first.")
            sys.exit(1)
        logger.info("Recomputing pr_size_loc on existing checkpoint...")
        # Load human PR data to refetch stats
        human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
        repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet")
        if "repo_id" not in human_pr_df.columns:
            url_to_id = dict(zip(repo_df["url"], repo_df["id"]))
            human_pr_df["repo_id"] = human_pr_df["repo_url"].map(url_to_id)
        # Fetch/load human PR stats (lightweight: 1 API call per PR)
        human_pr_stats_df = ingestor.fetch_human_pr_stats_only(human_pr_df)
        # Load AI commit details (human stats come from pr_stats_df directly)
        pr_commit_details_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
        # Recompute LOC
        ingestor.pr_size_LOC(pr_commit_details_df, human_pr_stats_df)
        # Also recompute fix_size since it depends on pr_size_loc
        if "fix_size" in ingestor.main_frame.columns:
            ingestor.main_frame.drop(columns=["fix_size"], inplace=True)
        is_fix = ingestor.main_frame["task_type"].fillna("").str.lower().str.strip() == "fix"
        ingestor.main_frame["fix_size"] = None
        if "pr_size_loc" in ingestor.main_frame.columns:
            ingestor.main_frame.loc[is_fix, "fix_size"] = ingestor.main_frame.loc[is_fix, "pr_size_loc"]
        ingestor.save_checkpoint()
        ingestor.save_final()
        logger.info("pr_size_loc recomputed and saved. Done.")
        sys.exit(0)

    if args.resume and ingestor.load_checkpoint():
        logger.info("Resumed from checkpoint – skipping data loading & early transforms.")
    else:
        # --------------------------------------------------------------
        # 1. Load datasets
        # --------------------------------------------------------------
        logger.info("Loading datasets from HuggingFace (AIDev-pop subset)...")

        # AIDev-pop: popular repos (>100 stars) – commit/review data keys to these
        ai_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
        repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet")
        user_df = pd.read_parquet("hf://datasets/hao-li/AIDev/user.parquet")

        # Also load all_user for broader coverage (pop user.parquet is smaller)
        all_user_df = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")

        # Comments and reviews (keyed to pull_request.parquet IDs)
        pr_reviews_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet")

        # Commits (keyed to pull_request.parquet IDs)
        pr_commits_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commits.parquet")
        pr_commit_details_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")

        # Task type
        pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet")

        # Issues (for defect linkage)
        related_issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/related_issue.parquet")
        issue_df = pd.read_parquet("hf://datasets/hao-li/AIDev/issue.parquet")

        # Human-PR
        human_pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pull_request.parquet")
        human_pr_task_type_df = pd.read_parquet("hf://datasets/hao-li/AIDev/human_pr_task_type.parquet")

        logger.info("Datasets loaded.")

        # --------------------------------------------------------------
        # 2a. Map human PRs repo_url → repo_id via pop repo table
        # --------------------------------------------------------------
        if "repo_id" not in human_pr_df.columns:
            url_to_id = dict(zip(repo_df["url"], repo_df["id"]))
            human_pr_df["repo_id"] = human_pr_df["repo_url"].map(url_to_id)
            mapped = human_pr_df["repo_id"].notna().sum()
            logger.info("Mapped repo_url → repo_id for %d / %d human PRs", mapped, len(human_pr_df))

        # --------------------------------------------------------------
        # 2b. Fetch human PR commit/review data from GitHub API (cached)
        # --------------------------------------------------------------
        human_commits_df, human_commit_details_df, human_reviews_df, human_pr_stats_df = \
            ingestor.fetch_human_pr_details(human_pr_df)

        # Merge human commit/review data into the main tables
        pr_commits_df = pd.concat([pr_commits_df, human_commits_df], ignore_index=True)
        pr_commits_df.drop_duplicates(subset=["sha", "pr_id"], inplace=True)

        pr_commit_details_df = pd.concat([pr_commit_details_df, human_commit_details_df], ignore_index=True)
        pr_commit_details_df.drop_duplicates(subset=["sha", "pr_id"], inplace=True)

        pr_reviews_df = pd.concat([pr_reviews_df, human_reviews_df], ignore_index=True)

        logger.info("After merging human data: %d commits, %d commit_details, %d reviews",
                     len(pr_commits_df), len(pr_commit_details_df), len(pr_reviews_df))

        # --------------------------------------------------------------
        # 2c. Combine AI + Human PRs and task types
        # --------------------------------------------------------------
        combined_pr_df = pd.concat([ai_pr_df, human_pr_df], ignore_index=True)
        combined_pr_df.drop_duplicates(subset=["id"], inplace=True)
        logger.info("Combined PR frame: %d rows", len(combined_pr_df))

        combined_task_type_df = pd.concat([pr_task_type_df, human_pr_task_type_df], ignore_index=True)
        combined_task_type_df.drop_duplicates(subset=["id"], inplace=True)
        logger.info("Combined task-type frame: %d rows", len(combined_task_type_df))

        # --------------------------------------------------------------
        # 3. Limit to target_count repos – skip already explored
        # --------------------------------------------------------------
        already_done = set(ingestor.explored_repos.keys())
        repo_id_to_name = dict(zip(repo_df["id"], repo_df["full_name"]))

        unique_repo_ids = combined_pr_df["repo_id"].dropna().unique()
        selected_repo_ids = []
        for rid in unique_repo_ids:
            name = repo_id_to_name.get(rid)
            if name and name not in already_done:
                selected_repo_ids.append(rid)
            if args.target_count is not None and len(selected_repo_ids) + len(already_done) >= args.target_count:
                break

        # Include previously explored repos so frame is complete
        explored_ids = [rid for rid, name in repo_id_to_name.items() if name in already_done]
        all_target_ids = set(selected_repo_ids) | set(explored_ids)

        logger.info("Target repos: %d new + %d explored = %d total (limit %s)",
                     len(selected_repo_ids), len(explored_ids), len(all_target_ids),
                     args.target_count if args.target_count is not None else "all")

        combined_pr_df = combined_pr_df[combined_pr_df["repo_id"].isin(all_target_ids)].copy()
        logger.info("PRs for target repos: %d", len(combined_pr_df))

        # --------------------------------------------------------------
        # 4. Run transformation pipeline with checkpointing
        # --------------------------------------------------------------
        try:
            ingestor.check_stars(repo_df, combined_pr_df)
            ingestor.save_checkpoint()

            if not args.skip_age:
                ingestor.check_age()
                ingestor.save_checkpoint()
            else:
                logger.info("Skipping check_age (--skip-age flag).")

            # Merge pop user_df + all_user_df for broadest coverage
            combined_user_df = pd.concat([user_df, all_user_df], ignore_index=True)
            combined_user_df.drop_duplicates(subset=["id"], inplace=True)

            ingestor.check_contributors(combined_user_df)
            ingestor.save_checkpoint()

            ingestor.check_ai_prs(combined_user_df)
            ingestor.save_checkpoint()

            ingestor.check_task_type(combined_task_type_df)
            ingestor.save_checkpoint()

            ingestor.check_domain_type()
            ingestor.language_type()
            ingestor.save_checkpoint()

            ingestor.time_to_first_review(pr_reviews_df)
            ingestor.save_checkpoint()

            ingestor.time_to_resolution()
            ingestor.save_checkpoint()

            ingestor.pr_size_LOC(pr_commit_details_df, human_pr_stats_df)
            ingestor.save_checkpoint()

            # Build combined file data for defect detection
            ai_files = pr_commit_details_df[["pr_id", "filename"]].dropna(subset=["filename"]).drop_duplicates()
            human_files = ingestor.fetch_human_pr_files(human_pr_df)
            pr_files_df = pd.concat([ai_files, human_files], ignore_index=True)
            pr_files_df = pr_files_df.dropna(subset=["filename"]).drop_duplicates(subset=["pr_id", "filename"])

            ingestor.defect_density(pr_files_df, related_issue_df, issue_df)
            ingestor.save_checkpoint()

            ingestor.fix_resolution_time()
            ingestor.fix_size(pr_commit_details_df)
            ingestor.fix_iteration_count(pr_commits_df)
            ingestor.save_checkpoint()

        except KeyboardInterrupt:
            logger.warning("Interrupted! Saving checkpoint...")
            ingestor.save_checkpoint()
            sys.exit(1)
        except Exception:
            logger.exception("Error during pipeline. Saving checkpoint...")
            ingestor.save_checkpoint()
            raise

    # ------------------------------------------------------------------
    # 5. Save final output
    # ------------------------------------------------------------------
    ingestor.save_final()

    if ingestor.main_frame is not None:
        mf = ingestor.main_frame
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("Total PRs       : %d", len(mf))
        logger.info("Unique repos    : %d", mf["repo_full_name"].nunique() if "repo_full_name" in mf.columns else 0)
        logger.info("AI PRs          : %d", int((mf["ai_pr"] == 1).sum()) if "ai_pr" in mf.columns else 0)
        logger.info("Human PRs       : %d", int((mf["ai_pr"] == 0).sum()) if "ai_pr" in mf.columns else 0)
        logger.info("Columns         : %s", list(mf.columns))
        logger.info("=" * 60)


if __name__ == "__main__":
    main()