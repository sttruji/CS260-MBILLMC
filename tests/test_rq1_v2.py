import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rq1_v2


def git(repo_path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FakeDatasetLoader:
    def __init__(self, ai_commits: pd.DataFrame, related_issues: pd.DataFrame = None, issues: pd.DataFrame = None):
        self.ai_commits = ai_commits
        self.related_issues = related_issues if related_issues is not None else pd.DataFrame(columns=["pr_id", "issue_id"])
        self.issues = issues if issues is not None else pd.DataFrame(columns=["id", "created_at", "closed_at", "updated_at"])

    def load_ai_pr_commits(self, pr_ids):
        return self.ai_commits[self.ai_commits["pr_id"].isin(pr_ids)].copy()

    def load_related_issues(self, pr_ids):
        return self.related_issues[self.related_issues["pr_id"].isin(pr_ids)].copy()

    def load_issues(self, issue_ids):
        if self.issues.empty:
            return self.issues.copy()
        return self.issues[self.issues["id"].isin(issue_ids)].copy()


class StaticRepoManager:
    def __init__(self, repo_mapping):
        self.repo_mapping = repo_mapping
        self.git_helper = rq1_v2.GitRepoManager(Path(tempfile.mkdtemp()))

    def ensure_repo(self, repo_full_name, refresh=False):
        return self.repo_mapping[repo_full_name]

    def existing_commits(self, repo_path, shas):
        return self.git_helper.existing_commits(repo_path, shas)

    def commit_dates(self, repo_path, shas):
        return self.git_helper.commit_dates(repo_path, shas)


class FakeSZZRunner:
    def __init__(self, repo_pairs):
        self.repo_pairs = repo_pairs

    def run(self, repo_full_name, issue_list_path, repo_path, refresh=False):
        pair_path = rq1_v2.SZZ_RESULT_DIR / rq1_v2.safe_repo_slug(repo_full_name) / "results" / "fix_and_introducers_pairs.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        with pair_path.open("w", encoding="utf-8") as handle:
            json.dump(self.repo_pairs.get(repo_full_name, []), handle)
        return pair_path


class DummyIngestor:
    def _github_get(self, url):
        raise AssertionError(f"Unexpected GitHub API call in test: {url}")

    def fetch_human_pr_details(self, human_pr_df):
        return (
            pd.DataFrame(columns=["pr_id", "sha"]),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


class RQ1V2LogicTests(unittest.TestCase):
    def test_candidate_fix_reasons_from_task_type_only(self):
        reasons = rq1_v2.candidate_fix_reasons("fix", "Refactor module naming", "", pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertEqual(reasons, ["task_type"])

    def test_candidate_fix_reasons_detects_revert(self):
        reasons = rq1_v2.candidate_fix_reasons("", 'Revert "introduce cache"', "", pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertIn("revert", reasons)

    def test_candidate_fix_reasons_detects_bugfix_text(self):
        reasons = rq1_v2.candidate_fix_reasons("", "Patch regression in parser", "", pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertIn("bugfix_text", reasons)

    def test_candidate_fix_reasons_detects_issue_closing(self):
        reasons = rq1_v2.candidate_fix_reasons("", "Resolve #123 parser failure", "", pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertIn("closing_issue_text", reasons)

    def test_candidate_fix_reasons_ignores_plain_issue_reference(self):
        reasons = rq1_v2.candidate_fix_reasons("", "Follow-up for #123 docs", "", pd.Timestamp("2025-01-01", tz="UTC"))
        self.assertEqual(reasons, [])

    def test_resolve_landed_commit_shas_filters_missing_shas(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp) / "repo"
            repo_path.mkdir()
            git(repo_path, "init")
            git(repo_path, "config", "user.email", "dev@example.com")
            git(repo_path, "config", "user.name", "Dev")
            write_text(repo_path / "app.py", "print('one')\n")
            git(repo_path, "add", "app.py")
            git(repo_path, "commit", "-m", "initial")
            head_sha = git(repo_path, "rev-parse", "HEAD")

            pr_commits = pd.DataFrame(
                [
                    {"pr_id": 1, "sha": head_sha},
                    {"pr_id": 1, "sha": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"},
                ]
            )
            merge_meta = pd.DataFrame([{"pr_id": 1, "merge_commit_sha": None}])
            manager = rq1_v2.GitRepoManager(Path(tmp) / "cache")
            landed = rq1_v2.resolve_landed_commit_shas(manager, repo_path, pr_commits, merge_meta)
            self.assertEqual(landed[1], {head_sha})

    def test_build_pair_trace_deduplicates_and_filters_invalid_links(self):
        repo_frame = pd.DataFrame(
            [
                {
                    "id": 1,
                    "repo_full_name": "owner/repo",
                    "created_at": "2025-01-01T00:00:00Z",
                    "merged_at": "2025-01-02T00:00:00Z",
                    "szz_candidate_fix": 0,
                    "szz_candidate_reason": "",
                },
                {
                    "id": 2,
                    "repo_full_name": "owner/repo",
                    "created_at": "2025-01-05T00:00:00Z",
                    "merged_at": "2025-01-06T00:00:00Z",
                    "szz_candidate_fix": 1,
                    "szz_candidate_reason": "bugfix_text",
                },
            ]
        )
        landed = {1: {"intro"}, 2: {"fix"}}
        with tempfile.TemporaryDirectory() as tmp:
            pair_path = Path(tmp) / "pairs.json"
            pair_path.write_text(json.dumps([["intro", "fix"], ["intro", "fix"], ["fix", "fix"]]), encoding="utf-8")
            trace = rq1_v2.build_pair_trace(repo_frame, landed, pair_path)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace.iloc[0]["origin_pr_id"], 1)
        self.assertEqual(trace.iloc[0]["fix_pr_id"], 2)

    def test_apply_defect_metrics_populates_windows_and_density(self):
        main_frame = pd.DataFrame(
            [
                {"id": 1, "pr_size_loc": 20},
                {"id": 2, "pr_size_loc": 0},
            ]
        )
        pair_trace = pd.DataFrame(
            [
                {
                    "origin_pr_id": 1,
                    "fix_pr_id": 10,
                    "origin_merged_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                    "fix_created_at": pd.Timestamp("2025-01-10T00:00:00Z"),
                },
                {
                    "origin_pr_id": 1,
                    "fix_pr_id": 11,
                    "origin_merged_at": pd.Timestamp("2025-01-01T00:00:00Z"),
                    "fix_created_at": pd.Timestamp("2025-02-15T00:00:00Z"),
                },
            ]
        )
        result = rq1_v2.apply_defect_metrics(main_frame, pair_trace)
        self.assertEqual(result.loc[result["id"] == 1, "defect_count_30d"].iloc[0], 1)
        self.assertEqual(result.loc[result["id"] == 1, "defect_count_60d"].iloc[0], 2)
        self.assertEqual(result.loc[result["id"] == 1, "defect_count_90d"].iloc[0], 2)
        self.assertEqual(result.loc[result["id"] == 1, "defect_density_90d"].iloc[0], 0.1)
        self.assertTrue(pd.isna(result.loc[result["id"] == 2, "defect_density_90d"].iloc[0]))
        self.assertEqual(result.loc[result["id"] == 1, "defect_density"].iloc[0], result.loc[result["id"] == 1, "defect_density_90d"].iloc[0])

    def test_build_issue_list_uses_issue_dates_with_pr_fallback(self):
        candidate_frame = pd.DataFrame(
            [
                {
                    "id": 2,
                    "created_at": pd.Timestamp("2025-01-05T00:00:00Z"),
                    "merged_at": pd.Timestamp("2025-01-06T00:00:00Z"),
                }
            ]
        )
        landed = {2: {"fix"}}
        commit_dates = {"fix": pd.Timestamp("2025-01-05T12:00:00Z")}
        issue_dates = {
            2: (
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-04T00:00:00Z"),
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "issue_list.json"
            payload = rq1_v2.build_issue_list(candidate_frame, landed, commit_dates, issue_dates, output_path)
            self.assertIn("PR-2-fix", payload)
            self.assertEqual(payload["PR-2-fix"]["creationdate"], "2025-01-01 00:00:00 +0000")


class RQ1V2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.results_dir = self.root / "results"
        self.processed_dir = self.root / "data" / "processed"
        self.cache_dir = self.root / "data" / "cache"
        self.repo_cache_dir = self.cache_dir / "repos"
        self.issue_list_dir = self.cache_dir / "szz_issue_lists"
        self.szz_result_dir = self.cache_dir / "szz_results"
        for path in [self.results_dir, self.processed_dir, self.repo_cache_dir, self.issue_list_dir, self.szz_result_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.original_paths = {
            "MERGE_METADATA_PATH": rq1_v2.MERGE_METADATA_PATH,
            "SZZ_ISSUE_LIST_DIR": rq1_v2.SZZ_ISSUE_LIST_DIR,
            "SZZ_RESULT_DIR": rq1_v2.SZZ_RESULT_DIR,
        }
        rq1_v2.MERGE_METADATA_PATH = self.processed_dir / "pr_merge_metadata.parquet"
        rq1_v2.SZZ_ISSUE_LIST_DIR = self.issue_list_dir
        rq1_v2.SZZ_RESULT_DIR = self.szz_result_dir
        self.addCleanup(self.restore_module_paths)

    def restore_module_paths(self):
        for name, value in self.original_paths.items():
            setattr(rq1_v2, name, value)

    def create_repo(self):
        repo_path = self.root / "repo"
        repo_path.mkdir()
        git(repo_path, "init")
        git(repo_path, "config", "user.email", "dev@example.com")
        git(repo_path, "config", "user.name", "Dev")

        write_text(repo_path / "app.py", "value = 1\n")
        git(repo_path, "add", "app.py")
        git(repo_path, "commit", "-m", "introduce bug")
        intro_sha = git(repo_path, "rev-parse", "HEAD")

        write_text(repo_path / "app.py", "value = 2\n")
        git(repo_path, "add", "app.py")
        git(repo_path, "commit", "-m", "fix parser regression")
        fix_sha = git(repo_path, "rev-parse", "HEAD")

        write_text(repo_path / "docs.md", "Some docs\n")
        git(repo_path, "add", "docs.md")
        git(repo_path, "commit", "-m", "follow-up docs")
        neutral_sha = git(repo_path, "rev-parse", "HEAD")

        return repo_path, intro_sha, fix_sha, neutral_sha

    def test_pipeline_smoke_path_preserves_rows_and_replaces_90d_metrics(self):
        repo_path, intro_sha, fix_sha, neutral_sha = self.create_repo()
        input_path = self.results_dir / "rq1_main_frame.parquet"
        output_path = self.results_dir / "rq1_main_frame_v2.parquet"
        output_csv = self.results_dir / "rq1_main_frame_v2.csv"
        pairs_path = self.results_dir / "rq1_szz_pairs.parquet"
        checkpoint_path = self.processed_dir / "rq1_v2_checkpoint.parquet"

        main_frame = pd.DataFrame(
            [
                {
                    "id": 1,
                    "number": 1,
                    "title": "Introduce parsing bug",
                    "body": "",
                    "agent": "Claude_Code",
                    "user_id": 1,
                    "user": "dev",
                    "state": "closed",
                    "created_at": "2025-01-01T00:00:00Z",
                    "closed_at": "2025-01-02T00:00:00Z",
                    "merged_at": "2025-01-02T00:00:00Z",
                    "repo_id": 1,
                    "repo_url": "https://api.github.com/repos/owner/repo",
                    "html_url": "https://github.com/owner/repo/pull/1",
                    "repo_api_url": "https://api.github.com/repos/owner/repo",
                    "repo_license": "MIT",
                    "repo_full_name": "owner/repo",
                    "repo_language": "Python",
                    "repo_forks": 100,
                    "repo_stars": 1000,
                    "star_group": 1,
                    "repo_created_at": "2020-01-01T00:00:00Z",
                    "repo_age_years": 5.0,
                    "age_group": 3,
                    "contributor_count": 10,
                    "contributor_group": 1,
                    "ai_pr": 1,
                    "user_followers": 0,
                    "user_following": 0,
                    "user_created_at": "2020-01-01T00:00:00Z",
                    "task_type": "feat",
                    "task_type_group": 0,
                    "domain_type_group": 0,
                    "language_type_group": 0,
                    "first_review_at": pd.NA,
                    "time_to_first_review": pd.NA,
                    "time_to_resolution": 24.0,
                    "pr_outcome": "accepted",
                    "fix_resolution_time": pd.NA,
                    "fix_iteration_count": 1,
                    "pr_size_loc": 20,
                    "pr_additions": 10,
                    "pr_deletions": 10,
                    "fix_size": pd.NA,
                    "defect_count_90d": 99,
                    "has_defect_90d": 1,
                    "defect_density": 4.95,
                },
                {
                    "id": 2,
                    "number": 2,
                    "title": "Fix parser regression",
                    "body": "Fixes #12",
                    "agent": "Claude_Code",
                    "user_id": 1,
                    "user": "dev",
                    "state": "closed",
                    "created_at": "2025-01-10T00:00:00Z",
                    "closed_at": "2025-01-11T00:00:00Z",
                    "merged_at": "2025-01-11T00:00:00Z",
                    "repo_id": 1,
                    "repo_url": "https://api.github.com/repos/owner/repo",
                    "html_url": "https://github.com/owner/repo/pull/2",
                    "repo_api_url": "https://api.github.com/repos/owner/repo",
                    "repo_license": "MIT",
                    "repo_full_name": "owner/repo",
                    "repo_language": "Python",
                    "repo_forks": 100,
                    "repo_stars": 1000,
                    "star_group": 1,
                    "repo_created_at": "2020-01-01T00:00:00Z",
                    "repo_age_years": 5.0,
                    "age_group": 3,
                    "contributor_count": 10,
                    "contributor_group": 1,
                    "ai_pr": 1,
                    "user_followers": 0,
                    "user_following": 0,
                    "user_created_at": "2020-01-01T00:00:00Z",
                    "task_type": "docs",
                    "task_type_group": 2,
                    "domain_type_group": 0,
                    "language_type_group": 0,
                    "first_review_at": pd.NA,
                    "time_to_first_review": pd.NA,
                    "time_to_resolution": 24.0,
                    "pr_outcome": "accepted",
                    "fix_resolution_time": 24.0,
                    "fix_iteration_count": 1,
                    "pr_size_loc": 10,
                    "pr_additions": 5,
                    "pr_deletions": 5,
                    "fix_size": 10,
                    "defect_count_90d": 0,
                    "has_defect_90d": 0,
                    "defect_density": 0.0,
                },
                {
                    "id": 3,
                    "number": 3,
                    "title": "Follow-up docs",
                    "body": "See #12",
                    "agent": "Claude_Code",
                    "user_id": 1,
                    "user": "dev",
                    "state": "closed",
                    "created_at": "2025-01-20T00:00:00Z",
                    "closed_at": "2025-01-21T00:00:00Z",
                    "merged_at": "2025-01-21T00:00:00Z",
                    "repo_id": 1,
                    "repo_url": "https://api.github.com/repos/owner/repo",
                    "html_url": "https://github.com/owner/repo/pull/3",
                    "repo_api_url": "https://api.github.com/repos/owner/repo",
                    "repo_license": "MIT",
                    "repo_full_name": "owner/repo",
                    "repo_language": "Python",
                    "repo_forks": 100,
                    "repo_stars": 1000,
                    "star_group": 1,
                    "repo_created_at": "2020-01-01T00:00:00Z",
                    "repo_age_years": 5.0,
                    "age_group": 3,
                    "contributor_count": 10,
                    "contributor_group": 1,
                    "ai_pr": 1,
                    "user_followers": 0,
                    "user_following": 0,
                    "user_created_at": "2020-01-01T00:00:00Z",
                    "task_type": "docs",
                    "task_type_group": 2,
                    "domain_type_group": 0,
                    "language_type_group": 0,
                    "first_review_at": pd.NA,
                    "time_to_first_review": pd.NA,
                    "time_to_resolution": 24.0,
                    "pr_outcome": "accepted",
                    "fix_resolution_time": pd.NA,
                    "fix_iteration_count": 1,
                    "pr_size_loc": 5,
                    "pr_additions": 5,
                    "pr_deletions": 0,
                    "fix_size": pd.NA,
                    "defect_count_90d": 0,
                    "has_defect_90d": 0,
                    "defect_density": 0.0,
                },
            ]
        )
        main_frame.to_parquet(input_path, index=False)

        merge_meta = pd.DataFrame(
            [
                {"pr_id": 1, "repo_full_name": "owner/repo", "number": 1, "merge_commit_sha": intro_sha, "merged_at": "2025-01-02T00:00:00Z", "state": "closed", "owner": "owner", "repo": "repo", "html_url": "https://github.com/owner/repo/pull/1"},
                {"pr_id": 2, "repo_full_name": "owner/repo", "number": 2, "merge_commit_sha": fix_sha, "merged_at": "2025-01-11T00:00:00Z", "state": "closed", "owner": "owner", "repo": "repo", "html_url": "https://github.com/owner/repo/pull/2"},
                {"pr_id": 3, "repo_full_name": "owner/repo", "number": 3, "merge_commit_sha": neutral_sha, "merged_at": "2025-01-21T00:00:00Z", "state": "closed", "owner": "owner", "repo": "repo", "html_url": "https://github.com/owner/repo/pull/3"},
            ]
        )
        merge_meta.to_parquet(rq1_v2.MERGE_METADATA_PATH, index=False)

        ai_commits = pd.DataFrame(
            [
                {"pr_id": 1, "sha": intro_sha},
                {"pr_id": 2, "sha": fix_sha},
                {"pr_id": 3, "sha": neutral_sha},
            ]
        )
        repo_manager = StaticRepoManager({"owner/repo": repo_path})
        szz_runner = FakeSZZRunner({"owner/repo": [[intro_sha, fix_sha]]})
        dataset_loader = FakeDatasetLoader(ai_commits=ai_commits)
        config = rq1_v2.PipelineConfig(
            input_path=input_path,
            output_path=output_path,
            output_csv_path=output_csv,
            pairs_path=pairs_path,
            checkpoint_path=checkpoint_path,
            repo_filter="owner/repo",
            limit_repos=1,
        )
        pipeline = rq1_v2.RQ1V2Pipeline(
            config=config,
            dataset_loader=dataset_loader,
            repo_manager=repo_manager,
            szz_runner=szz_runner,
            ingestor=DummyIngestor(),
        )

        final_frame, pair_trace = pipeline.run()
        self.assertEqual(len(final_frame), 3)
        self.assertEqual(len(pair_trace), 1)
        self.assertTrue(output_path.exists())
        self.assertTrue(output_csv.exists())
        self.assertTrue(pairs_path.exists())
        self.assertTrue(checkpoint_path.exists())

        origin_row = final_frame[final_frame["id"] == 1].iloc[0]
        fix_row = final_frame[final_frame["id"] == 2].iloc[0]
        neutral_row = final_frame[final_frame["id"] == 3].iloc[0]

        self.assertEqual(origin_row["defect_count_30d"], 1)
        self.assertEqual(origin_row["defect_count_90d"], 1)
        self.assertEqual(origin_row["has_defect_90d"], 1)
        self.assertEqual(origin_row["defect_density"], origin_row["defect_density_90d"])
        self.assertEqual(fix_row["szz_candidate_fix"], 1)
        self.assertIn("bugfix_text", fix_row["szz_candidate_reason"])
        self.assertEqual(neutral_row["szz_candidate_fix"], 0)

    @unittest.skipUnless(os.getenv("RQ1_V2_RUN_PIPELINE_INTEGRATION") == "1", "opt-in integration test")
    def test_opt_in_pipeline_integration_only_bugfix_candidate_generates_pair(self):
        repo_path, intro_sha, fix_sha, neutral_sha = self.create_repo()
        repo_frame = pd.DataFrame(
            [
                {"id": 1, "repo_full_name": "owner/repo", "created_at": "2025-01-01T00:00:00Z", "merged_at": "2025-01-02T00:00:00Z", "title": "Introduce parsing bug", "body": "", "task_type": "feat", "ai_pr": 1, "html_url": "https://github.com/owner/repo/pull/1", "number": 1, "pr_size_loc": 20},
                {"id": 2, "repo_full_name": "owner/repo", "created_at": "2025-01-10T00:00:00Z", "merged_at": "2025-01-11T00:00:00Z", "title": "Fix parser regression", "body": "Fixes #12", "task_type": "docs", "ai_pr": 1, "html_url": "https://github.com/owner/repo/pull/2", "number": 2, "pr_size_loc": 10},
                {"id": 3, "repo_full_name": "owner/repo", "created_at": "2025-01-20T00:00:00Z", "merged_at": "2025-01-21T00:00:00Z", "title": "Follow-up docs", "body": "See #12", "task_type": "docs", "ai_pr": 1, "html_url": "https://github.com/owner/repo/pull/3", "number": 3, "pr_size_loc": 5},
            ]
        )
        annotated = rq1_v2.annotate_candidate_fix_columns(repo_frame)
        ai_commits = pd.DataFrame(
            [
                {"pr_id": 1, "sha": intro_sha},
                {"pr_id": 2, "sha": fix_sha},
                {"pr_id": 3, "sha": neutral_sha},
            ]
        )
        merge_meta = pd.DataFrame(
            [
                {"pr_id": 1, "merge_commit_sha": intro_sha},
                {"pr_id": 2, "merge_commit_sha": fix_sha},
                {"pr_id": 3, "merge_commit_sha": neutral_sha},
            ]
        )
        repo_manager = StaticRepoManager({"owner/repo": repo_path})
        landed = rq1_v2.resolve_landed_commit_shas(repo_manager, repo_path, ai_commits, merge_meta)
        pair_dir = self.root / "pairs.json"
        pair_dir.write_text(json.dumps([[intro_sha, fix_sha], [intro_sha, neutral_sha]]), encoding="utf-8")
        trace = rq1_v2.build_pair_trace(annotated, landed, pair_dir)
        self.assertEqual(set(trace["fix_pr_id"]), {2})


if __name__ == "__main__":
    unittest.main()
