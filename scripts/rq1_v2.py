from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rq1_ingest import BASE_DIR as INGEST_BASE_DIR  # noqa: E402
from rq1_ingest import RepoIngestor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(INGEST_BASE_DIR)
DEFAULT_INPUT_PATH = BASE_DIR / "results" / "rq1_main_frame.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "results" / "rq1_main_frame_v2.parquet"
DEFAULT_OUTPUT_CSV_PATH = BASE_DIR / "results" / "rq1_main_frame_v2.csv"
DEFAULT_PAIRS_PATH = BASE_DIR / "results" / "rq1_szz_pairs.parquet"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / "rq1_v2_checkpoint.parquet"
MERGE_METADATA_PATH = BASE_DIR / "data" / "processed" / "pr_merge_metadata.parquet"

DATA_CACHE_DIR = BASE_DIR / "data" / "cache"
REPO_CACHE_DIR = DATA_CACHE_DIR / "repos"
SZZ_TOOL_DIR = DATA_CACHE_DIR / "tools" / "SZZUnleashed"
SZZ_ISSUE_LIST_DIR = DATA_CACHE_DIR / "szz_issue_lists"
SZZ_RESULT_DIR = DATA_CACHE_DIR / "szz_results"

HF_PR_COMMITS = "hf://datasets/hao-li/AIDev/pr_commits.parquet"
HF_RELATED_ISSUES = "hf://datasets/hao-li/AIDev/related_issue.parquet"
HF_ISSUES = "hf://datasets/hao-li/AIDev/issue.parquet"

BUGFIX_RE = re.compile(r"\b(fix|fixes|fixed|bugfix|hotfix|defect|regression|repair|resolve|resolved|patch)\b", re.I)
REVERT_RE = re.compile(r"\b(revert|reverted|rollback|rolled back)\b", re.I)
CLOSING_HASH_RE = re.compile(r"\b(fix(e[sd])?|close[sd]?|resolve[sd]?)\s+#\d+\b", re.I)
CLOSING_URL_RE = re.compile(
    r"\b(fix(e[sd])?|close[sd]?|resolve[sd]?)\s+https://github\.com/[^/\s]+/[^/\s]+/issues/\d+\b",
    re.I,
)

DEFAULT_CHECKPOINT_EVERY = 10
SZZ_GRADLE_IMAGE = "gradle:6.9.4-jdk8"


class RepoUnavailableError(Exception):
    """Raised when a repository cannot be cloned (DMCA, deleted, private, etc.)"""
    pass


class SZZTimeoutError(Exception):
    """Raised when SZZ takes longer than the configured timeout."""
    pass


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    output_csv_path: Path
    pairs_path: Path
    checkpoint_path: Path
    resume: bool = False
    repo_filter: Optional[str] = None
    limit_repos: Optional[int] = None
    refresh_repos: bool = False
    refresh_szz: bool = False
    cleanup_repos: bool = False
    checkpoint_every: int = 10
    szz_timeout: Optional[int] = None  # timeout in minutes for SZZ per repo


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_repo_slug(repo_full_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_full_name)


def pipe_join(values: Sequence[str]) -> str:
    return "|".join(value for value in values if value)


def run_command(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ, **(dict(env) if env else {})),
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        cmd = " ".join(args)
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        raise RuntimeError(f"Command failed: {cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return completed


def parse_html_url(html_url: str) -> Tuple[str, str, int]:
    parts = str(html_url or "").rstrip("/").split("/")
    if len(parts) < 5:
        raise ValueError(f"Cannot parse html_url: {html_url}")
    owner, repo, pr_number = parts[-4], parts[-3], int(parts[-1])
    return owner, repo, pr_number


def normalize_datetimes(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)


def format_szz_timestamp(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S %z")


def candidate_fix_reasons(task_type: object, title: object, body: object, merged_at: object) -> List[str]:
    if pd.isna(merged_at):
        return []

    reasons: List[str] = []
    task = str(task_type or "").strip().lower()
    text = " ".join(part for part in [str(title or ""), str(body or "")] if part).strip()

    if task == "fix":
        reasons.append("task_type")
    if REVERT_RE.search(text):
        reasons.append("revert")
    if BUGFIX_RE.search(text):
        reasons.append("bugfix_text")
    if CLOSING_HASH_RE.search(text) or CLOSING_URL_RE.search(text):
        reasons.append("closing_issue_text")
    return reasons


def apply_local_filters(frame: pd.DataFrame, filters: Sequence[Tuple[str, str, object]]) -> pd.DataFrame:
    filtered = frame
    for column, operation, value in filters:
        if column not in filtered.columns:
            continue
        if operation == "in":
            filtered = filtered[filtered[column].isin(list(value))]
        elif operation in ("=", "=="):
            filtered = filtered[filtered[column] == value]
        else:
            raise ValueError(f"Unsupported filter operation: {operation}")
    return filtered


def read_parquet_with_fallback(
    path_or_uri: str,
    columns: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[Tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    try:
        return pd.read_parquet(path_or_uri, columns=list(columns) if columns else None, filters=filters)
    except Exception:
        frame = pd.read_parquet(path_or_uri, columns=list(columns) if columns else None)
        return apply_local_filters(frame, filters or [])


class AIDevDatasetLoader:
    def load_ai_pr_commits(self, pr_ids: Sequence[int]) -> pd.DataFrame:
        if not pr_ids:
            return pd.DataFrame(columns=["pr_id", "sha"])
        frame = read_parquet_with_fallback(
            HF_PR_COMMITS,
            columns=["pr_id", "sha"],
            filters=[("pr_id", "in", list(pr_ids))],
        )
        return frame[["pr_id", "sha"]].dropna(subset=["pr_id", "sha"]).drop_duplicates()

    def load_related_issues(self, pr_ids: Sequence[int]) -> pd.DataFrame:
        if not pr_ids:
            return pd.DataFrame(columns=["pr_id", "issue_id"])
        frame = read_parquet_with_fallback(
            HF_RELATED_ISSUES,
            columns=["pr_id", "issue_id"],
            filters=[("pr_id", "in", list(pr_ids))],
        )
        return frame[["pr_id", "issue_id"]].dropna(subset=["pr_id", "issue_id"]).drop_duplicates()

    def load_issues(self, issue_ids: Sequence[int]) -> pd.DataFrame:
        if not issue_ids:
            return pd.DataFrame(columns=["id", "created_at", "closed_at", "updated_at"])
        preferred_columns = ["id", "created_at", "closed_at", "updated_at"]
        try:
            frame = read_parquet_with_fallback(
                HF_ISSUES,
                columns=preferred_columns,
                filters=[("id", "in", list(issue_ids))],
            )
        except Exception:
            frame = read_parquet_with_fallback(
                HF_ISSUES,
                filters=[("id", "in", list(issue_ids))],
            )
        available = [column for column in preferred_columns if column in frame.columns]
        return frame[available].dropna(subset=["id"]).drop_duplicates(subset=["id"])


class GitRepoManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        ensure_directory(cache_dir)

    def repo_path(self, repo_full_name: str) -> Path:
        return self.cache_dir / safe_repo_slug(repo_full_name)

    def ensure_repo(self, repo_full_name: str, refresh: bool = False) -> Path:
        repo_path = self.repo_path(repo_full_name)
        # Skip LFS files - we only need commit history for SZZ analysis
        git_env = {"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
        if (repo_path / ".git").exists():
            if refresh:
                logger.info("Refreshing repo cache for %s", repo_full_name)
                run_command(["git", "-C", str(repo_path), "fetch", "--all", "--tags", "--prune"], env=git_env)
            return repo_path

        clone_url = f"https://github.com/{repo_full_name}.git"
        ensure_directory(repo_path.parent)
        logger.info("Cloning %s into %s", repo_full_name, repo_path)
        # Disable LFS filter to avoid issues with repos that use git-lfs
        try:
            run_command([
                "git", "clone",
                "-c", "filter.lfs.smudge=",
                "-c", "filter.lfs.process=",
                "-c", "filter.lfs.required=false",
                clone_url, str(repo_path)
            ], env=git_env)
        except RuntimeError as e:
            # Clean up partial clone if it exists
            if repo_path.exists():
                shutil.rmtree(repo_path)
            # Check for common unavailable repo errors
            error_msg = str(e).lower()
            if any(phrase in error_msg for phrase in [
                "dmca", "404", "403", "not found", "repository unavailable",
                "could not read from remote", "does not exist", "unable to access"
            ]):
                raise RepoUnavailableError(f"Repository unavailable: {repo_full_name}") from e
            raise
        return repo_path

    def existing_commits(self, repo_path: Path, shas: Iterable[str]) -> Set[str]:
        unique_shas = sorted({sha for sha in shas if sha})
        if not unique_shas:
            return set()
        batch_input = "\n".join(unique_shas) + "\n"
        result = run_command(
            ["git", "-C", str(repo_path), "cat-file", "--batch-check=%(objectname) %(objecttype)"],
            input_text=batch_input,
        )
        existing: Set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "commit":
                existing.add(parts[0])
        return existing

    def commit_dates(self, repo_path: Path, shas: Iterable[str]) -> Dict[str, pd.Timestamp]:
        unique_shas = sorted({sha for sha in shas if sha})
        if not unique_shas:
            return {}

        commit_dates: Dict[str, pd.Timestamp] = {}
        batch_size = 200
        for start in range(0, len(unique_shas), batch_size):
            batch = unique_shas[start : start + batch_size]
            result = run_command(
                ["git", "-C", str(repo_path), "log", "--no-walk", "--format=%H%x09%cI", *batch]
            )
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                sha, timestamp = line.split("\t", 1)
                commit_dates[sha] = pd.to_datetime(timestamp, utc=True)
        return commit_dates


class SZZUnleashedRunner:
    def __init__(self, tool_dir: Path):
        self.tool_dir = tool_dir
        ensure_directory(tool_dir.parent)

    @property
    def jar_override(self) -> Optional[Path]:
        override = os.getenv("RQ1_V2_SZZ_JAR")
        if not override:
            return None
        jar_path = Path(override).expanduser().resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"RQ1_V2_SZZ_JAR points to missing file: {jar_path}")
        return jar_path

    def ensure_source(self, refresh: bool = False) -> Path:
        git_env = {"GIT_TERMINAL_PROMPT": "0"}
        if (self.tool_dir / ".git").exists():
            if refresh:
                logger.info("Refreshing SZZUnleashed source in %s", self.tool_dir)
                run_command(["git", "-C", str(self.tool_dir), "pull", "--ff-only"], env=git_env)
            return self.tool_dir

        logger.info("Cloning SZZUnleashed into %s", self.tool_dir)
        run_command(
            ["git", "clone", "https://github.com/wogscpar/SZZUnleashed.git", str(self.tool_dir)],
            env=git_env,
        )
        return self.tool_dir

    def ensure_jar(self, refresh: bool = False) -> Path:
        override = self.jar_override
        if override:
            return override

        source_dir = self.ensure_source(refresh=refresh)
        jar_candidates = sorted((source_dir / "szz" / "build" / "libs").glob("szz_find_bug_introducers-*.jar"))
        if jar_candidates and not refresh:
            return jar_candidates[-1]

        logger.info("Building SZZUnleashed fat jar with Docker")
        mount_path = str(source_dir.resolve())
        run_command(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{mount_path}:/work",
                "-w",
                "/work/szz",
                SZZ_GRADLE_IMAGE,
                "sh",
                "-lc",
                "gradle build && gradle fatJar",
            ]
        )
        jar_candidates = sorted((source_dir / "szz" / "build" / "libs").glob("szz_find_bug_introducers-*.jar"))
        if not jar_candidates:
            raise FileNotFoundError("SZZUnleashed build completed but no jar was produced.")
        return jar_candidates[-1]

    def result_path_for_repo(self, repo_full_name: str) -> Path:
        return SZZ_RESULT_DIR / safe_repo_slug(repo_full_name) / "results" / "fix_and_introducers_pairs.json"

    def partitioned_result_dir(self, repo_full_name: str) -> Path:
        """Path to the issues/ directory where partitioned input files are stored."""
        return SZZ_RESULT_DIR / safe_repo_slug(repo_full_name) / "issues"

    def results_dir(self, repo_full_name: str) -> Path:
        """Path to the results/ directory."""
        return SZZ_RESULT_DIR / safe_repo_slug(repo_full_name) / "results"

    def szz_ran_successfully(self, repo_full_name: str) -> bool:
        """
        Check if SZZ has already run for this repo (even if it found no pairs).
        SZZ creates the issues/ directory with partitioned input files when it runs,
        and outputs results to results/ or results/result*/.
        """
        # Main result file exists
        if self.result_path_for_repo(repo_full_name).exists():
            return True
        # Check for partitioned output results in results/result*/
        results_dir = self.results_dir(repo_full_name)
        if results_dir.exists():
            partition_results = list(results_dir.glob("result*/fix_and_introducers_pairs.json"))
            if partition_results:
                return True
        # Check for issues/ directory (means SZZ ran, possibly found nothing)
        issues_dir = self.partitioned_result_dir(repo_full_name)
        if issues_dir.exists() and list(issues_dir.glob("fix_and_introducers_pairs_*.json")):
            return True
        return False

    def merge_partitioned_results(self, repo_full_name: str) -> Optional[Path]:
        """
        Merge SZZ results into a single file.
        
        SZZ outputs results in two possible layouts:
        1. Single file: results/fix_and_introducers_pairs.json
        2. Partitioned: results/result0/, results/result1/, ... each with fix_and_introducers_pairs.json
        
        If partitioned, merge all partition outputs into the main result file.
        If SZZ ran but found no pairs (issues/ exists but no results), create empty file.
        """
        pair_path = self.result_path_for_repo(repo_full_name)
        if pair_path.exists():
            return pair_path

        # Check for partitioned output results in results/result*/
        results_dir = self.results_dir(repo_full_name)
        if results_dir.exists():
            partition_files = sorted(results_dir.glob("result*/fix_and_introducers_pairs.json"))
            if partition_files:
                logger.info("Merging %d partitioned SZZ output files for %s", len(partition_files), repo_full_name)
                merged_pairs: list = []
                for part_file in partition_files:
                    try:
                        with part_file.open("r", encoding="utf-8") as handle:
                            data = json.load(handle)
                        if isinstance(data, list):
                            merged_pairs.extend(data)
                        elif isinstance(data, dict):
                            for record in data.values():
                                if isinstance(record, (list, tuple)) and len(record) >= 2:
                                    merged_pairs.append([str(record[0]), str(record[1])])
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", part_file, exc)
                ensure_directory(pair_path.parent)
                with pair_path.open("w", encoding="utf-8") as handle:
                    json.dump(merged_pairs, handle)
                logger.info("Merged SZZ results: %d pairs → %s", len(merged_pairs), pair_path)
                return pair_path

        # SZZ ran (issues/ exists) but found no pairs at all
        issues_dir = self.partitioned_result_dir(repo_full_name)
        if issues_dir.exists() and list(issues_dir.glob("fix_and_introducers_pairs_*.json")):
            logger.info("SZZ ran for %s but found no bug-introducing pairs, creating empty result", repo_full_name)
            ensure_directory(pair_path.parent)
            with pair_path.open("w", encoding="utf-8") as handle:
                json.dump([], handle)
            return pair_path

        return None

    def has_cached_result(self, repo_full_name: str) -> bool:
        """Check if SZZ results already exist for this repo (for resume logic)."""
        return self.szz_ran_successfully(repo_full_name)

    def get_or_merge_results(self, repo_full_name: str) -> Path:
        """Get the result path, merging partitioned results if necessary."""
        pair_path = self.result_path_for_repo(repo_full_name)
        if pair_path.exists():
            return pair_path
        # Try to merge partitioned results
        merged = self.merge_partitioned_results(repo_full_name)
        if merged:
            return merged
        raise FileNotFoundError(f"No SZZ results found for {repo_full_name}")

    def run(
        self,
        repo_full_name: str,
        issue_list_path: Path,
        repo_path: Path,
        refresh: bool = False,
        timeout_minutes: Optional[int] = None,
    ) -> Path:
        pair_path = self.result_path_for_repo(repo_full_name)
        if pair_path.exists() and not refresh:
            return pair_path
        
        # Check for partitioned results that can be merged
        if not refresh and self.has_cached_result(repo_full_name):
            return self.get_or_merge_results(repo_full_name)

        jar_path = self.ensure_jar(refresh=refresh)
        work_dir = pair_path.parent.parent
        ensure_directory(work_dir)
        if pair_path.parent.exists():
            shutil.rmtree(pair_path.parent)
        # Also clear issues dir if refreshing
        issues_dir = self.partitioned_result_dir(repo_full_name)
        if issues_dir.exists():
            shutil.rmtree(issues_dir)

        logger.info("Running SZZ Phase 2 for %s", repo_full_name)
        timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
        cmd = [
            "java",
            "-jar",
            str(jar_path),
            "-i",
            str(issue_list_path),
            "-r",
            str(repo_path),
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                logger.warning("SZZ returned non-zero for %s: %s", repo_full_name, stderr[:500])
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning("SZZ timed out after %d minutes for %s, collecting partial results",
                          timeout_minutes, repo_full_name)
        
        # Check for result - either single file or partitioned
        if pair_path.exists():
            return pair_path
        
        # Try to merge partitioned results (works for both complete and partial/timed-out runs)
        merged = self.merge_partitioned_results(repo_full_name)
        if merged:
            if timed_out:
                logger.info("Collected partial SZZ results for %s despite timeout", repo_full_name)
            return merged

        if timed_out:
            raise SZZTimeoutError(f"SZZ timed out after {timeout_minutes}min for {repo_full_name} with no usable results")
            
        raise FileNotFoundError(f"SZZUnleashed did not produce {pair_path}")


def fetch_human_pr_commits(ingestor: RepoIngestor, frame: pd.DataFrame) -> pd.DataFrame:
    human_subset = frame.loc[frame["ai_pr"] == 0, ["id", "html_url"]].drop_duplicates(subset=["id"])
    if human_subset.empty:
        return pd.DataFrame(columns=["pr_id", "sha"])

    commits_df, _, _, _ = ingestor.fetch_human_pr_details(human_subset)
    if commits_df.empty:
        return pd.DataFrame(columns=["pr_id", "sha"])
    return commits_df[["pr_id", "sha"]].dropna(subset=["pr_id", "sha"]).drop_duplicates()


def fetch_merge_metadata(
    ingestor: RepoIngestor,
    frame: pd.DataFrame,
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    cached_ids = set(cached["pr_id"].tolist()) if not cached.empty and not refresh else set()

    merged = frame.loc[frame["merged_at"].notna(), ["id", "html_url", "repo_full_name", "number", "merged_at"]]
    missing = merged[~merged["id"].isin(cached_ids)]
    rows: List[Dict[str, object]] = []

    logger.info("PR merge metadata: cached=%d missing=%d", len(cached_ids), len(missing))
    for _, row in missing.iterrows():
        pr_id = int(row["id"])
        owner, repo, pr_number = parse_html_url(row["html_url"])
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        payload = ingestor._github_get(url)
        if isinstance(payload, dict):
            rows.append(
                {
                    "pr_id": pr_id,
                    "repo_full_name": f"{owner}/{repo}",
                    "number": int(pr_number),
                    "merge_commit_sha": payload.get("merge_commit_sha"),
                    "merged_at": payload.get("merged_at") or row.get("merged_at"),
                    "state": payload.get("state"),
                    "owner": owner,
                    "repo": repo,
                    "html_url": row["html_url"],
                }
            )
        else:
            rows.append(
                {
                    "pr_id": pr_id,
                    "repo_full_name": row["repo_full_name"],
                    "number": int(pr_number),
                    "merge_commit_sha": None,
                    "merged_at": row["merged_at"],
                    "state": None,
                    "owner": owner,
                    "repo": repo,
                    "html_url": row["html_url"],
                }
            )

    merged_cache = cached if not refresh else pd.DataFrame()
    if rows:
        new_frame = pd.DataFrame(rows)
        merged_cache = pd.concat([merged_cache, new_frame], ignore_index=True)
    if merged_cache.empty:
        merged_cache = pd.DataFrame(
            columns=["pr_id", "repo_full_name", "number", "merge_commit_sha", "merged_at", "state", "owner", "repo", "html_url"]
        )
    merged_cache.drop_duplicates(subset=["pr_id"], keep="last", inplace=True)
    normalize_datetimes(merged_cache, ["merged_at"])
    ensure_directory(cache_path.parent)
    merged_cache.to_parquet(cache_path, index=False)
    return merged_cache


def resolve_landed_commit_shas(
    repo_manager: GitRepoManager,
    repo_path: Path,
    pr_commits_df: pd.DataFrame,
    merge_meta_df: pd.DataFrame,
) -> Dict[int, Set[str]]:
    pr_to_shas: Dict[int, Set[str]] = {}
    for pr_id, group in pr_commits_df.groupby("pr_id"):
        pr_to_shas[int(pr_id)] = set(group["sha"].dropna())

    for row in merge_meta_df.itertuples(index=False):
        pr_id = int(row.pr_id)
        merge_sha = getattr(row, "merge_commit_sha", None)
        if merge_sha:
            pr_to_shas.setdefault(pr_id, set()).add(merge_sha)

    all_shas = {sha for shas in pr_to_shas.values() for sha in shas}
    existing = repo_manager.existing_commits(repo_path, all_shas)
    return {pr_id: {sha for sha in shas if sha in existing} for pr_id, shas in pr_to_shas.items()}


def build_issue_date_lookup(
    candidate_frame: pd.DataFrame,
    related_issue_df: pd.DataFrame,
    issue_df: pd.DataFrame,
) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    normalize_datetimes(issue_df, [column for column in ["created_at", "closed_at", "updated_at"] if column in issue_df.columns])

    pr_to_issue_ids: Dict[int, Set[int]] = {}
    if not related_issue_df.empty:
        for row in related_issue_df.itertuples(index=False):
            pr_to_issue_ids.setdefault(int(row.pr_id), set()).add(int(row.issue_id))

    issue_lookup = issue_df.set_index("id", drop=False) if not issue_df.empty else pd.DataFrame()
    date_lookup: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}

    for row in candidate_frame.itertuples(index=False):
        pr_id = int(row.id)
        creation_candidates: List[pd.Timestamp] = []
        resolution_candidates: List[pd.Timestamp] = []
        for issue_id in pr_to_issue_ids.get(pr_id, set()):
            if issue_lookup.empty or issue_id not in issue_lookup.index:
                continue
            issue_row = issue_lookup.loc[issue_id]
            created_at = issue_row["created_at"] if "created_at" in issue_row else pd.NaT
            closed_at = issue_row["closed_at"] if "closed_at" in issue_row else pd.NaT
            updated_at = issue_row["updated_at"] if "updated_at" in issue_row else pd.NaT
            if pd.notna(created_at):
                creation_candidates.append(created_at)
            if pd.notna(closed_at):
                resolution_candidates.append(closed_at)
            elif pd.notna(updated_at):
                resolution_candidates.append(updated_at)

        fallback_creation = pd.to_datetime(row.created_at, utc=True)
        fallback_resolution = pd.to_datetime(row.merged_at, utc=True)
        creation = min(creation_candidates) if creation_candidates else fallback_creation
        resolution = max(resolution_candidates) if resolution_candidates else fallback_resolution
        date_lookup[pr_id] = (creation, resolution)

    return date_lookup


def build_issue_list(
    candidate_frame: pd.DataFrame,
    landed_shas: Dict[int, Set[str]],
    commit_dates: Dict[str, pd.Timestamp],
    issue_dates: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]],
    output_path: Path,
) -> Dict[str, Dict[str, str]]:
    payload: Dict[str, Dict[str, str]] = {}
    for row in candidate_frame.itertuples(index=False):
        pr_id = int(row.id)
        creation_date, resolution_date = issue_dates[pr_id]
        fix_shas = sorted(landed_shas.get(pr_id, set()))
        for sha in fix_shas:
            commit_date = commit_dates.get(sha, pd.to_datetime(row.merged_at, utc=True))
            key = f"PR-{pr_id}-{sha}"
            payload[key] = {
                "creationdate": format_szz_timestamp(creation_date),
                "resolutiondate": format_szz_timestamp(resolution_date),
                "hash": sha,
                "commitdate": format_szz_timestamp(commit_date),
            }

    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def extract_pair_values(record: object) -> Optional[Tuple[str, str]]:
    """Return (bug_introducing_sha, fixing_sha) from an SZZ output record.

    SZZUnleashed outputs list pairs as [fixing_sha, bug_introducing_sha]
    (see examples/BugIntroducersFinder.md: "the first in each pair is the
    fixing commits and the second is the bug introducing commit").
    We normalise to (bug_introducing, fixing) for downstream use.
    """
    if isinstance(record, (list, tuple)) and len(record) >= 2:
        # SZZUnleashed list format: [fixing, introducing] → swap
        return str(record[1]), str(record[0])

    if isinstance(record, dict):
        introducing_keys = [
            "bugIntroducingCommit",
            "bugIntroducingCommitSha",
            "bug_introducing_commit",
            "bugIntroducingRevision",
            "bic",
            "introducing",
        ]
        fixing_keys = [
            "fixCommit",
            "fixCommitSha",
            "bugFixingCommit",
            "fixingCommit",
            "bug_fixing_commit",
            "fix",
        ]
        introducing_sha = next((record.get(key) for key in introducing_keys if record.get(key)), None)
        fixing_sha = next((record.get(key) for key in fixing_keys if record.get(key)), None)
        if introducing_sha and fixing_sha:
            return str(introducing_sha), str(fixing_sha)
    return None


def load_szz_pairs(pair_path: Path) -> List[Tuple[str, str]]:
    with pair_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: List[Tuple[str, str]] = []
    if isinstance(payload, list):
        for record in payload:
            values = extract_pair_values(record)
            if values:
                records.append(values)
    elif isinstance(payload, dict):
        for record in payload.values():
            values = extract_pair_values(record)
            if values:
                records.append(values)
    return records


def build_pair_trace(
    repo_frame: pd.DataFrame,
    landed_shas: Dict[int, Set[str]],
    pair_path: Path,
) -> pd.DataFrame:
    szz_pairs = load_szz_pairs(pair_path)
    if not szz_pairs:
        return pd.DataFrame(
            columns=[
                "repo_full_name",
                "origin_pr_id",
                "fix_pr_id",
                "bug_introducing_sha",
                "fixing_sha",
                "szz_candidate_reason",
                "origin_merged_at",
                "fix_created_at",
            ]
        )

    normalize_datetimes(repo_frame, ["created_at", "merged_at"])
    repo_full_name = repo_frame["repo_full_name"].iloc[0]

    origin_lookup: Dict[str, Set[int]] = {}
    fix_lookup: Dict[str, Set[int]] = {}
    reason_lookup: Dict[int, str] = {}
    created_lookup: Dict[int, pd.Timestamp] = {}
    merged_lookup: Dict[int, pd.Timestamp] = {}
    merged_pr_ids = set(repo_frame.loc[repo_frame["merged_at"].notna(), "id"].astype(int))
    candidate_fix_ids = set(repo_frame.loc[repo_frame["szz_candidate_fix"] == 1, "id"].astype(int))

    for row in repo_frame.itertuples(index=False):
        pr_id = int(row.id)
        for sha in landed_shas.get(pr_id, set()):
            if pr_id in merged_pr_ids:
                origin_lookup.setdefault(sha, set()).add(pr_id)
            if pr_id in candidate_fix_ids:
                fix_lookup.setdefault(sha, set()).add(pr_id)
        reason_lookup[pr_id] = str(row.szz_candidate_reason or "")
        created_lookup[pr_id] = pd.to_datetime(row.created_at, utc=True)
        merged_lookup[pr_id] = pd.to_datetime(row.merged_at, utc=True)

    trace_rows: List[Dict[str, object]] = []
    for bug_introducing_sha, fixing_sha in szz_pairs:
        origin_ids = origin_lookup.get(bug_introducing_sha, set())
        fix_ids = fix_lookup.get(fixing_sha, set())
        if not origin_ids or not fix_ids:
            continue
        for origin_pr_id in origin_ids:
            for fix_pr_id in fix_ids:
                if origin_pr_id == fix_pr_id:
                    continue
                fix_created_at = created_lookup.get(fix_pr_id)
                origin_merged_at = merged_lookup.get(origin_pr_id)
                if pd.isna(fix_created_at) or pd.isna(origin_merged_at):
                    continue
                if fix_created_at <= origin_merged_at:
                    continue
                trace_rows.append(
                    {
                        "repo_full_name": repo_full_name,
                        "origin_pr_id": origin_pr_id,
                        "fix_pr_id": fix_pr_id,
                        "bug_introducing_sha": bug_introducing_sha,
                        "fixing_sha": fixing_sha,
                        "szz_candidate_reason": reason_lookup.get(fix_pr_id, ""),
                        "origin_merged_at": origin_merged_at,
                        "fix_created_at": fix_created_at,
                    }
                )

    trace = pd.DataFrame(trace_rows)
    if trace.empty:
        return trace
    trace.drop_duplicates(
        subset=["origin_pr_id", "fix_pr_id", "bug_introducing_sha", "fixing_sha"],
        inplace=True,
    )
    return trace


def compute_density(counts: pd.Series, loc: pd.Series) -> pd.Series:
    density = counts.astype(float) / loc.astype(float)
    density[(loc.isna()) | (loc <= 0)] = pd.NA
    return density


def apply_defect_metrics(main_frame: pd.DataFrame, pair_trace: pd.DataFrame) -> pd.DataFrame:
    result = main_frame.copy()
    windows = [30, 60, 90]

    for window in windows:
        result[f"defect_count_{window}d"] = 0
        result[f"has_defect_{window}d"] = 0
        result[f"defect_density_{window}d"] = pd.NA

    if not pair_trace.empty:
        pair_window = pair_trace[["origin_pr_id", "fix_pr_id", "origin_merged_at", "fix_created_at"]].drop_duplicates()
        pair_window["delta_days"] = (
            pair_window["fix_created_at"] - pair_window["origin_merged_at"]
        ).dt.total_seconds() / (24 * 3600)

        for window in windows:
            eligible = pair_window[(pair_window["delta_days"] > 0) & (pair_window["delta_days"] <= window)]
            counts = eligible.groupby("origin_pr_id")["fix_pr_id"].nunique()
            result[f"defect_count_{window}d"] = result["id"].map(counts).fillna(0).astype(int)
            result[f"has_defect_{window}d"] = (result[f"defect_count_{window}d"] > 0).astype(int)
            result[f"defect_density_{window}d"] = compute_density(result[f"defect_count_{window}d"], result["pr_size_loc"])
    else:
        for window in windows:
            result[f"defect_density_{window}d"] = compute_density(result[f"defect_count_{window}d"], result["pr_size_loc"])

    result["defect_density"] = result["defect_density_90d"]
    return result


def annotate_candidate_fix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    reasons = annotated.apply(
        lambda row: candidate_fix_reasons(row.get("task_type"), row.get("title"), row.get("body"), row.get("merged_at")),
        axis=1,
    )
    annotated["szz_candidate_reason"] = reasons.apply(pipe_join)
    annotated["szz_candidate_fix"] = reasons.apply(lambda value: int(bool(value)))
    if "szz_observable" not in annotated.columns:
        annotated["szz_observable"] = 0
    annotated["szz_observable"] = annotated["szz_observable"].fillna(0).astype(int)
    return annotated


class RQ1V2Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        dataset_loader: Optional[AIDevDatasetLoader] = None,
        repo_manager: Optional[GitRepoManager] = None,
        szz_runner: Optional[SZZUnleashedRunner] = None,
        ingestor: Optional[RepoIngestor] = None,
    ):
        self.config = config
        self.dataset_loader = dataset_loader or AIDevDatasetLoader()
        self.repo_manager = repo_manager or GitRepoManager(REPO_CACHE_DIR)
        self.szz_runner = szz_runner or SZZUnleashedRunner(SZZ_TOOL_DIR)
        self.ingestor = ingestor or RepoIngestor(target_count=None)

    def load_starting_frame(self) -> pd.DataFrame:
        frame = pd.read_parquet(self.config.input_path)
        normalize_datetimes(frame, ["created_at", "closed_at", "merged_at", "repo_created_at", "user_created_at"])

        if self.config.repo_filter:
            frame = frame[frame["repo_full_name"] == self.config.repo_filter].copy()
        elif self.config.limit_repos:
            selected_repos = sorted(frame["repo_full_name"].dropna().unique())[: self.config.limit_repos]
            frame = frame[frame["repo_full_name"].isin(selected_repos)].copy()

        if self.config.resume and self.config.checkpoint_path.exists():
            checkpoint = pd.read_parquet(self.config.checkpoint_path)
            normalize_datetimes(checkpoint, ["created_at", "closed_at", "merged_at", "repo_created_at", "user_created_at"])
            if set(checkpoint["id"]) == set(frame["id"]):
                checkpoint = checkpoint.set_index("id")
                frame = frame.set_index("id")
                for column in [
                    "szz_observable",
                    "szz_candidate_fix",
                    "szz_candidate_reason",
                    "defect_count_30d",
                    "has_defect_30d",
                    "defect_density_30d",
                    "defect_count_60d",
                    "has_defect_60d",
                    "defect_density_60d",
                    "defect_count_90d",
                    "has_defect_90d",
                    "defect_density_90d",
                    "defect_density",
                ]:
                    if column in checkpoint.columns:
                        frame[column] = checkpoint[column]
                frame = frame.reset_index()
                logger.info("Loaded resumable columns from %s", self.config.checkpoint_path)
            else:
                logger.info("Ignoring checkpoint because it does not match the current input cohort.")

        return annotate_candidate_fix_columns(frame)

    def save_checkpoint(self, frame: pd.DataFrame) -> None:
        ensure_directory(self.config.checkpoint_path.parent)
        frame.to_parquet(self.config.checkpoint_path, index=False)

    def save_outputs(self, frame: pd.DataFrame, pair_trace: pd.DataFrame) -> None:
        ensure_directory(self.config.output_path.parent)
        frame.to_parquet(self.config.output_path, index=False)
        frame.to_csv(self.config.output_csv_path, index=False)
        pair_trace.to_parquet(self.config.pairs_path, index=False)

    def load_commit_shas(self, frame: pd.DataFrame) -> pd.DataFrame:
        ai_pr_ids = frame.loc[frame["ai_pr"] == 1, "id"].astype(int).tolist()
        ai_commits = self.dataset_loader.load_ai_pr_commits(ai_pr_ids)
        human_commits = fetch_human_pr_commits(self.ingestor, frame)
        combined = pd.concat([ai_commits, human_commits], ignore_index=True)
        combined = combined.dropna(subset=["pr_id", "sha"]).drop_duplicates(subset=["pr_id", "sha"])
        return combined

    def load_issue_metadata(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        candidate_ids = frame.loc[frame["szz_candidate_fix"] == 1, "id"].astype(int).tolist()
        related_issue_df = self.dataset_loader.load_related_issues(candidate_ids)
        issue_ids = related_issue_df["issue_id"].dropna().astype(int).unique().tolist() if not related_issue_df.empty else []
        issue_df = self.dataset_loader.load_issues(issue_ids)
        return related_issue_df, issue_df

    def process_repo(
        self,
        repo_full_name: str,
        main_frame: pd.DataFrame,
        pr_commits_df: pd.DataFrame,
        merge_meta_df: pd.DataFrame,
        related_issue_df: pd.DataFrame,
        issue_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        repo_path = self.repo_manager.ensure_repo(repo_full_name, refresh=self.config.refresh_repos)
        repo_frame = main_frame[main_frame["repo_full_name"] == repo_full_name].copy()
        repo_pr_ids = repo_frame["id"].astype(int).tolist()
        repo_commits = pr_commits_df[pr_commits_df["pr_id"].isin(repo_pr_ids)].copy()
        repo_merge_meta = merge_meta_df[merge_meta_df["pr_id"].isin(repo_pr_ids)].copy()

        landed_shas = resolve_landed_commit_shas(self.repo_manager, repo_path, repo_commits, repo_merge_meta)

        observable_ids = [
            pr_id
            for pr_id, shas in landed_shas.items()
            if shas and pr_id in set(repo_frame.loc[repo_frame["merged_at"].notna(), "id"].astype(int))
        ]
        if observable_ids:
            main_frame.loc[main_frame["id"].isin(observable_ids), "szz_observable"] = 1

        candidate_frame = repo_frame[(repo_frame["szz_candidate_fix"] == 1) & (repo_frame["merged_at"].notna())].copy()
        candidate_frame = candidate_frame[candidate_frame["id"].isin([pr_id for pr_id, shas in landed_shas.items() if shas])]
        if candidate_frame.empty:
            return main_frame, pd.DataFrame()

        issue_dates = build_issue_date_lookup(
            candidate_frame,
            related_issue_df[related_issue_df["pr_id"].isin(candidate_frame["id"])].copy() if not related_issue_df.empty else related_issue_df,
            issue_df,
        )
        fix_shas = {sha for pr_id in candidate_frame["id"].astype(int) for sha in landed_shas.get(pr_id, set())}
        commit_dates = self.repo_manager.commit_dates(repo_path, fix_shas)

        issue_list_path = SZZ_ISSUE_LIST_DIR / f"{safe_repo_slug(repo_full_name)}_issue_list.json"
        build_issue_list(candidate_frame, landed_shas, commit_dates, issue_dates, issue_list_path)

        pair_path = self.szz_runner.run(
            repo_full_name=repo_full_name,
            issue_list_path=issue_list_path,
            repo_path=repo_path,
            refresh=self.config.refresh_szz,
            timeout_minutes=self.config.szz_timeout,
        )
        pair_trace = build_pair_trace(repo_frame, landed_shas, pair_path)
        return main_frame, pair_trace

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ensure_directory(REPO_CACHE_DIR)
        ensure_directory(SZZ_ISSUE_LIST_DIR)
        ensure_directory(SZZ_RESULT_DIR)

        main_frame = self.load_starting_frame()
        self.save_checkpoint(main_frame)

        pr_commits_df = self.load_commit_shas(main_frame)
        merge_meta_df = fetch_merge_metadata(
            self.ingestor,
            main_frame,
            MERGE_METADATA_PATH,
            refresh=self.config.refresh_szz or self.config.refresh_repos,
        )
        related_issue_df, issue_df = self.load_issue_metadata(main_frame)

        repos = sorted(
            main_frame.loc[
                (main_frame["merged_at"].notna()) & (main_frame["szz_candidate_fix"] == 1),
                "repo_full_name",
            ].dropna().unique()
        )
        if self.config.limit_repos and not self.config.repo_filter:
            repos = repos[: self.config.limit_repos]

        pair_frames: List[pd.DataFrame] = []

        # When resuming, filter out repos that already have SZZ results
        if self.config.resume:
            already_processed = [r for r in repos if self.szz_runner.has_cached_result(r)]
            repos_to_process = [r for r in repos if not self.szz_runner.has_cached_result(r)]
            logger.info("Resume mode: %d repos already have SZZ results, %d remaining to process",
                       len(already_processed), len(repos_to_process))
            # Load existing pair traces for already-processed repos
            for repo_full_name in already_processed:
                # Use get_or_merge_results to ensure the result file exists (creates empty file if SZZ found no pairs)
                try:
                    pair_path = self.szz_runner.get_or_merge_results(repo_full_name)
                except FileNotFoundError:
                    logger.warning("Could not load SZZ results for %s, will reprocess", repo_full_name)
                    repos_to_process.append(repo_full_name)
                    continue
                repo_frame = main_frame[main_frame["repo_full_name"] == repo_full_name].copy()
                repo_pr_ids = repo_frame["id"].astype(int).tolist()
                repo_commits = pr_commits_df[pr_commits_df["pr_id"].isin(repo_pr_ids)].copy()
                repo_merge_meta = merge_meta_df[merge_meta_df["pr_id"].isin(repo_pr_ids)].copy()
                repo_path = self.repo_manager.repo_path(repo_full_name)
                if repo_path.exists():
                    landed_shas = resolve_landed_commit_shas(self.repo_manager, repo_path, repo_commits, repo_merge_meta)
                else:
                    # Approximate: use commit SHAs from pr_commits_df directly
                    landed_shas = {int(pr_id): set(group["sha"].dropna()) for pr_id, group in repo_commits.groupby("pr_id")}
                pair_trace = build_pair_trace(repo_frame, landed_shas, pair_path)
                if not pair_trace.empty:
                    pair_frames.append(pair_trace)
        else:
            repos_to_process = list(repos)

        checkpoint_every = self.config.checkpoint_every
        skipped_repos: List[str] = []
        for index, repo_full_name in enumerate(repos_to_process, start=1):
            logger.info("Processing repo %d/%d: %s", index, len(repos_to_process), repo_full_name)
            try:
                main_frame, pair_trace = self.process_repo(
                    repo_full_name,
                    main_frame,
                    pr_commits_df,
                    merge_meta_df,
                    related_issue_df,
                    issue_df,
                )
                if not pair_trace.empty:
                    pair_frames.append(pair_trace)
            except RepoUnavailableError as e:
                logger.warning("Skipping unavailable repo %s: %s", repo_full_name, e)
                skipped_repos.append(repo_full_name)
                continue
            except SZZTimeoutError as e:
                logger.warning("SZZ timed out for %s: %s (partial results collected)", repo_full_name, e)
                # Partial results were already merged inside run(); try to use them
                try:
                    pair_path = self.szz_runner.get_or_merge_results(repo_full_name)
                    repo_frame = main_frame[main_frame["repo_full_name"] == repo_full_name].copy()
                    repo_pr_ids = repo_frame["id"].astype(int).tolist()
                    repo_commits = pr_commits_df[pr_commits_df["pr_id"].isin(repo_pr_ids)].copy()
                    repo_merge_meta = merge_meta_df[merge_meta_df["pr_id"].isin(repo_pr_ids)].copy()
                    repo_path = self.repo_manager.repo_path(repo_full_name)
                    if repo_path.exists():
                        landed_shas = resolve_landed_commit_shas(self.repo_manager, repo_path, repo_commits, repo_merge_meta)
                    else:
                        landed_shas = {int(pr_id): set(group["sha"].dropna()) for pr_id, group in repo_commits.groupby("pr_id")}
                    pair_trace = build_pair_trace(repo_frame, landed_shas, pair_path)
                    if not pair_trace.empty:
                        pair_frames.append(pair_trace)
                    logger.info("Collected partial SZZ results for %s", repo_full_name)
                except Exception as inner_exc:
                    logger.warning("Could not collect partial results for %s: %s", repo_full_name, inner_exc)
                    skipped_repos.append(repo_full_name)

            # Clean up cloned repo to save disk space
            if self.config.cleanup_repos:
                repo_path = self.repo_manager.repo_path(repo_full_name)
                if repo_path.exists():
                    logger.info("Cleaning up repo: %s", repo_path)
                    shutil.rmtree(repo_path)

            if index % checkpoint_every == 0:
                partial_pairs = (
                    pd.concat(pair_frames, ignore_index=True)
                    if pair_frames
                    else pd.DataFrame(
                        columns=[
                            "repo_full_name",
                            "origin_pr_id",
                            "fix_pr_id",
                            "bug_introducing_sha",
                            "fixing_sha",
                            "szz_candidate_reason",
                            "origin_merged_at",
                            "fix_created_at",
                        ]
                    )
                )
                self.save_checkpoint(apply_defect_metrics(main_frame, partial_pairs))
                logger.info("Checkpoint saved after %d repos", index)

        pair_trace = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(
            columns=[
                "repo_full_name",
                "origin_pr_id",
                "fix_pr_id",
                "bug_introducing_sha",
                "fixing_sha",
                "szz_candidate_reason",
                "origin_merged_at",
                "fix_created_at",
            ]
        )
        final_frame = apply_defect_metrics(main_frame, pair_trace)
        self.save_checkpoint(final_frame)
        self.save_outputs(final_frame, pair_trace)
        if skipped_repos:
            logger.warning("Skipped %d unavailable repos: %s", len(skipped_repos), ", ".join(skipped_repos))
        logger.info("rq1_v2 complete: %d rows, %d pair rows", len(final_frame), len(pair_trace))
        return final_frame, pair_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RQ1 v2 with SZZ-backed defect metrics.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input main frame parquet path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output parquet path.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV_PATH), help="Output CSV path.")
    parser.add_argument("--resume", action="store_true", help="Reuse rq1_v2 checkpoint and cached repo results when available.")
    parser.add_argument("--repo", default=None, help="Process only one repo_full_name cohort.")
    parser.add_argument("--limit-repos", type=int, default=None, help="Limit processing to the first N repos in sorted order.")
    parser.add_argument("--refresh-repos", action="store_true", help="Refresh cached git repos before analysis.")
    parser.add_argument("--refresh-szz", action="store_true", help="Rebuild or rerun SZZ even when cached results exist.")
    parser.add_argument("--cleanup-repos", action="store_true", help="Delete cloned repos after processing to save disk space.")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, 
                       help=f"Save checkpoint after every N repos (default: {DEFAULT_CHECKPOINT_EVERY}). Use 1 for maximum safety.")
    parser.add_argument("--szz-timeout", type=int, default=None,
                       help="Timeout in minutes for SZZ per repo. Partial results are collected on timeout.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        output_csv_path=Path(args.output_csv).expanduser().resolve(),
        pairs_path=DEFAULT_PAIRS_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        resume=args.resume,
        repo_filter=args.repo,
        limit_repos=args.limit_repos,
        refresh_repos=args.refresh_repos,
        refresh_szz=args.refresh_szz,
        cleanup_repos=args.cleanup_repos,
        checkpoint_every=args.checkpoint_every,
        szz_timeout=args.szz_timeout,
    )
    pipeline = RQ1V2Pipeline(config=config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
