import argparse
import os
import sys
import logging
import re
import json
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
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
RQ1_MAIN_FRAME = os.path.join(BASE_DIR, "results", "rq1_main_frame.parquet")
AI_COMMIT_DETAILS_CACHE = os.path.join(BASE_DIR, "data", "processed", "ai_pr_commit_details.parquet")
DEFECT_PATTERNS_CACHE = os.path.join(BASE_DIR, "data", "processed", "rq3_defect_patterns.parquet")
COGNITIVE_LOAD_CACHE = os.path.join(BASE_DIR, "data", "processed", "rq3_cognitive_load.parquet")
LLM_PATTERNS_CACHE = os.path.join(BASE_DIR, "data", "processed", "rq3_llm_patterns.parquet")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "data", "processed", "rq3_checkpoint.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "rq3_defect_classifier_frame.parquet")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "results", "rq3_defect_classifier_frame.csv")

load_dotenv(os.path.join(BASE_DIR, ".env"))


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def ensure_ai_commit_details_cached():
    """Download and cache AI commit details from HuggingFace if not already cached."""
    if os.path.exists(AI_COMMIT_DETAILS_CACHE):
        logger.info("AI commit details cache found: %s", AI_COMMIT_DETAILS_CACHE)
        return
    
    logger.info("Downloading AI commit details from HuggingFace (first time only)...")
    try:
        df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet")
        logger.info("Downloaded %d rows, saving to cache...", len(df))
        df.to_parquet(AI_COMMIT_DETAILS_CACHE)
        logger.info("Cache saved successfully")
    except Exception as e:
        logger.error("Failed to download from HuggingFace: %s", e)
        raise


# ---------------------------------------------------------------------------
# ODC Schema Definition
# ---------------------------------------------------------------------------
"""
Orthogonal Defect Classification (ODC) Framework
================================================

ODC provides a multi-dimensional taxonomy for classifying defects based on:
1. Type: What kind of defect (API misuse, logic error, etc.)
2. Class: How the defect manifests structurally
3. Trigger: When the defect was discovered
4. Impact: Severity of the defect

References:
- Chillarege, R., et al. "Orthogonal defect classification-a concept for in-process 
  measurements." IEEE transactions on software engineering 18.11 (1992): 943-956.
"""

ODC_TYPE_GROUPS = {
    0: "API_MISUSE",           # Incorrect API call, wrong parameter order, signature mismatch
    1: "LOGIC",                # Boolean logic error, boundary condition, wrong operator
    2: "INTERFACE",            # Data format mismatch, incompatible interface usage
    3: "DATA",                 # Variable initialization, scope, lifetime, type issues
    4: "CHECKING",             # Insufficient validation, missing bounds check, null check
    5: "DOCUMENTATION",        # Comments, error messages, docstrings, type annotations
    6: "ASSIGNMENT",           # Variable assignment, operator precedence, side effects
    7: "TIMING",               # Race condition, deadlock, synchronization, order dependency
    8: "BUILD_INTEGRATION",    # Build system, configuration, dependencies, imports
    9: "OTHER"                 # Doesn't fit above categories
}

ODC_CLASS_GROUPS = {
    0: "FUNCTION",             # Logic within a single function/method
    1: "INTERFACE",            # Interaction between functions/modules
    2: "MODULE",               # Structure/organization within module
    3: "BUILD_INTEGRATION",    # Integration across modules/system
    4: "MISSING_FEATURE",      # Incomplete implementation or missing functionality
    5: "DOCUMENTATION",        # Documentation/comments issues
    6: "OTHER"                 # Other structural issues
}

ODC_TRIGGER_GROUPS = {
    0: "CODE_REVIEW",          # Found during peer code review
    1: "UNIT_TEST",            # Unit test failure
    2: "INTEGRATION_TEST",     # Integration test failure
    3: "SYSTEM_TEST",          # System/acceptance test
    4: "FIELD_TEST",           # Production/field use (customer report)
    5: "DESIGN_REVIEW",        # Design review finding
    6: "OTHER"                 # Other discovery context
}

ODC_IMPACT_GROUPS = {
    0: "CRITICAL",             # System down, data loss, security breach
    1: "MAJOR",                # Significant functionality broken
    2: "MODERATE",             # Feature degradation, workaround available
    3: "MINOR",                # Cosmetic or rare edge case
    4: "UNKNOWN"               # Unable to determine severity
}

# Cognitive Load Theory categories
CLT_CATEGORIES = {
    0: "LOW",                  # Total load < 2.0
    1: "MODERATE",             # Total load 2.0-3.9
    2: "HIGH",                 # Total load 4.0-5.9
    3: "VERY_HIGH"             # Total load >= 6.0 (overload risk)
}


class DefectClassifier:
    """
    Multi-stage defect classification system mapping commits to ODC taxonomy.
    
    Pipeline stages:
    1. Extract code patterns from patches (static analysis)
    2. Compute cognitive load metrics (CLT framework)
    3. Detect LLM-specific error patterns (probabilistic logic signatures)
    4. Assign ODC classification based on patterns + context
    """
    
    def __init__(self):
        self.main_frame = None
        self.pr_commit_details_df = None
        self.patterns_df = None
        self.cognitive_load_df = None
        self.llm_patterns_df = None
        self.odc_df = None
        self.final_frame = None
        self.defect_patterns = {}  # sha -> pattern scores
        self.cognitive_load = {}   # sha -> CLT metrics
        self.llm_patterns = {}     # sha -> LLM error likelihood
        self.odc_classified = {}   # sha -> {odc_type, odc_class, odc_trigger, odc_impact}
        
    def load_rq1_data(self):
        """Load RQ1 main frame and commit details from cache and local data."""
        logger.info("Loading RQ1 main frame and commit details...")
        
        self.main_frame = pd.read_parquet(RQ1_MAIN_FRAME)
        
        # TEMPORARY: Limit to 200 AI + 200 human PRs for faster testing
        # ai_prs = self.main_frame[self.main_frame['ai_pr'] == 1].head(200)
        # human_prs = self.main_frame[self.main_frame['ai_pr'] == 0].head(200)
        # self.main_frame = pd.concat([ai_prs, human_prs], ignore_index=True)

        # Use full dataset (no PR limit)
        ai_prs = self.main_frame[self.main_frame['ai_pr'] == 1]
        human_prs = self.main_frame[self.main_frame['ai_pr'] == 0]
        self.main_frame = pd.concat([ai_prs, human_prs], ignore_index=True)
        
        logger.info("Loaded RQ1 main frame: %d rows", len(self.main_frame))
        
        # Ensure AI commit details are cached, then load from cache
        ensure_ai_commit_details_cached()
        logger.info("Loading AI commit details from cache...")
        self.pr_commit_details_df = pd.read_parquet(AI_COMMIT_DETAILS_CACHE)
        logger.info("Loaded AI commit details: %d rows", len(self.pr_commit_details_df))
        
        # Load and merge human PR commit details if available
        human_details_path = os.path.join(BASE_DIR, "data", "processed", "human_pr_commit_details.parquet")
        if os.path.exists(human_details_path):
            logger.info("Loading human commit details from %s...", human_details_path)
            human_details = pd.read_parquet(human_details_path)
            logger.info("Loaded human commit details: %d rows", len(human_details))
            
            # Combine both datasets
            self.pr_commit_details_df = pd.concat(
                [self.pr_commit_details_df, human_details], ignore_index=True
            )
            logger.info("Combined AI + Human commit details: %d rows total", len(self.pr_commit_details_df))
        else:
            logger.warning("Human commit details not found at %s - proceeding with AI-only data", human_details_path)
    
    def extract_code_patterns(self):
        """
        Phase 1: Extract code patterns from commit patches indicating defect types.
        
        Heuristic-based pattern detection analyzing:
        - API calls and parameter usage
        - Logic operators and conditions
        - Null/bounds checking
        - Timing synchronization primitives
        - Documentation and comments
        - Variable scope and initialization
        
        **COMMIT-LEVEL GRANULARITY**: One analysis per unique commit (sha), 
        aggregates multiple files affected within a single commit.
        """
        logger.info("Extracting code patterns from %d commits (commit-level)...", 
                   self.pr_commit_details_df['sha'].nunique())
        
        # Deduplicate on sha to get one row per commit
        unique_commits = self.pr_commit_details_df[['sha', 'pr_id', 'patch', 'filename']].drop_duplicates(subset=['sha'], keep='first')
        
        patterns = []
        for idx, (_, row) in enumerate(unique_commits.iterrows()):
            if idx % 10000 == 0:
                logger.info("  [%d/%d] patterns extracted...", idx, len(unique_commits))
            
            sha = row.get("sha")
            pr_id = row.get("pr_id")
            patch = str(row.get("patch", ""))
            filename = str(row.get("filename", ""))
            
            if not sha or not patch or patch == "nan":
                continue
            
            analysis = self._analyze_patch(patch, filename)
            analysis["sha"] = sha
            analysis["pr_id"] = pr_id
            patterns.append(analysis)
        
        self.patterns_df = pd.DataFrame(patterns)
        
        # Cache patterns
        os.makedirs(os.path.dirname(DEFECT_PATTERNS_CACHE), exist_ok=True)
        self.patterns_df.to_parquet(DEFECT_PATTERNS_CACHE, index=False)
        logger.info("Pattern extraction complete: %d patterns cached (commit-level).", len(self.patterns_df))
        
        return self.patterns_df
    
    def _analyze_patch(self, patch, filename):
        """
        Analyze single patch for code patterns indicative of defect types.
        
        Returns dict with scores [0, 1] for each ODC type likelihood.
        """
        removed_lines = [l for l in patch.split('\n') if l.startswith('-') and not l.startswith('---')]
        added_lines = [l for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++')]
        all_lines = removed_lines + added_lines
        
        # Basic metrics
        n_removed = len(removed_lines)
        n_added = len(added_lines)
        lines_changed = n_removed + n_added
        
        # API misuse patterns
        api_score = self._detect_api_patterns(removed_lines, added_lines)
        
        # Logic error patterns
        logic_score = self._detect_logic_errors(removed_lines, added_lines)
        
        # Interface/data mismatch
        interface_score = self._detect_interface_issues(removed_lines, added_lines)
        
        # Data/initialization issues
        data_score = self._detect_data_issues(removed_lines, added_lines)
        
        # Missing checks/validation
        checking_score = self._detect_missing_checks(removed_lines, added_lines)
        
        # Documentation issues
        doc_score = self._detect_doc_issues(removed_lines, added_lines)
        
        # Assignment/operator issues
        assignment_score = self._detect_assignment_issues(removed_lines, added_lines)
        
        # Timing/concurrency issues
        timing_score = self._detect_timing_issues(removed_lines, added_lines)
        
        # Build/integration issues
        build_score = self._detect_build_issues(removed_lines, added_lines, filename)
        
        # Code complexity indicators
        max_nesting = self._compute_max_nesting(all_lines)
        variable_count = len(set(re.findall(r'\b[a-zA-Z_]\w*\b', ' '.join(all_lines))))
        
        return {
            "api_misuse_score": api_score,
            "logic_error_score": logic_score,
            "interface_mismatch_score": interface_score,
            "data_error_score": data_score,
            "missing_checks_score": checking_score,
            "documentation_issue_score": doc_score,
            "assignment_error_score": assignment_score,
            "timing_issue_score": timing_score,
            "build_error_score": build_score,
            "lines_added": n_added,
            "lines_deleted": n_removed,
            "lines_changed": lines_changed,
            "max_nesting_depth": max_nesting,
            "variable_count": variable_count,
            "is_test_file": 1 if any(x in filename for x in ["test", "spec", "__test__"]) else 0,
            "is_doc_file": 1 if any(x in filename for x in ["README", ".md", "doc"]) else 0,
        }
    
    def _detect_api_patterns(self, removed_lines, added_lines):
        """Detect API misuse: wrong function names, parameter order, signatures."""
        api_keywords = [
            'import', 'from', 'class', 'def', 'function', 'return',
            'call', 'invoke', 'method', 'new', 'constructor'
        ]
        
        # Count API-related changes
        api_changes = sum(
            1 for l in added_lines + removed_lines
            if any(kw in l.lower() for kw in api_keywords) and 
            (l.count('(') != l.count(')') or re.search(r'\w+\s*\(', l))
        )
        
        # Detect common API misuse patterns
        common_mistakes = [
            (r'sort\([^,)]*,[^)]*\)', 'sort with args'),  # Wrong sort params
            (r'split\(["\']([^"\']+)["\'][^)]*\)', 'split with wrong sep'),
            (r'join\([^)]*\)', 'join usage'),
            (r'\.append\(.*\)', 'list append'),
        ]
        
        mistake_count = sum(
            len(re.findall(pattern, ' '.join(added_lines)))
            for pattern, _ in common_mistakes
        )
        
        score = min(1.0, (api_changes + mistake_count * 0.5) / max(len(added_lines), 1) * 2)
        return score
    
    def _detect_logic_errors(self, removed_lines, added_lines):
        """Detect logic errors: boolean operators, comparisons, conditions."""
        logic_operators = ['and', 'or', 'not', '==', '!=', '<', '>', '<=', '>=', 'if', 'else']
        
        # Count logic operator changes
        logic_changes = sum(
            1 for l in removed_lines + added_lines
            if any(op in l for op in logic_operators)
        )
        
        # Detect potential logic inversions
        inversions = sum(
            1 for i, l in enumerate(added_lines)
            if any(f'not {op}' in l for op in ['true', 'false', 'True', 'False']) and
            i > 0 and any(op in removed_lines[i] if i < len(removed_lines) else False 
                         for op in ['not'])
        )
        
        score = min(1.0, (logic_changes + inversions * 2) / max(len(added_lines), 1) * 0.3)
        return score
    
    def _detect_interface_issues(self, removed_lines, added_lines):
        """Detect interface mismatches: type changes, format issues, incompatibility."""
        interface_patterns = [
            r'\b(int|str|float|bool|list|dict|array|object|Any)\b',
            r'\.to_\w+\(', r'\.as_\w+\(', r'\.convert\(',
            r'JSON\.parse', r'JSON\.stringify',
            r'pickle\.load', r'pickle\.dump',
        ]
        
        interface_changes = sum(
            len(re.findall(pattern, ' '.join(added_lines + removed_lines)))
            for pattern in interface_patterns
        )
        
        # Type annotation changes
        type_changes = sum(1 for l in added_lines if ':' in l and '==' not in l)
        
        score = min(1.0, (interface_changes + type_changes) / max(len(added_lines), 1) * 0.2)
        return score
    
    def _detect_data_issues(self, removed_lines, added_lines):
        """Detect data issues: initialization, scope, lifetime problems."""
        init_patterns = [
            r'\b(let|var|const|int|float|str|bool|list|dict)\s+\w+\s*=',
            r'self\.\w+\s*=',
            r'this\.\w+\s*=',
        ]
        
        init_count = sum(
            len(re.findall(pattern, ' '.join(added_lines)))
            for pattern in init_patterns
        )
        
        # Detect uninitialized usage (risky pattern)
        uninitialized = sum(1 for l in added_lines if re.search(r'\b\w+\s*[+\-*/=]', l))
        
        # Scope markers
        scope_issues = sum(1 for l in added_lines + removed_lines if 'global' in l or 'nonlocal' in l)
        
        score = min(1.0, (init_count + uninitialized * 0.5 + scope_issues) / max(len(added_lines), 1) * 0.3)
        return score
    
    def _detect_missing_checks(self, removed_lines, added_lines):
        """Detect missing validation: null checks, bounds checks, error handling."""
        check_patterns = [
            r'if\s+\w+\s*(is\s+)?not\s+(None|null|undefined)',
            r'if\s+len\(',
            r'if\s+.*\s*in\s+',
            r'try:', r'except', r'catch',
            r'assert\s+',
            r'raise\s+', r'throw\s+',
        ]
        
        checks_added = sum(
            len(re.findall(pattern, ' '.join(added_lines)))
            for pattern in check_patterns
        )
        
        checks_removed = sum(
            len(re.findall(pattern, ' '.join(removed_lines)))
            for pattern in check_patterns
        )
        
        # If more removed than added, likely missing validation
        check_deficit = max(0, checks_removed - checks_added)
        
        score = min(1.0, check_deficit / max(len(added_lines), 1) * 2)
        return score
    
    def _detect_doc_issues(self, removed_lines, added_lines):
        """Detect documentation issues: comments, docstrings, type hints."""
        doc_patterns = [
            r'""".*?"""',  # Docstrings
            r"'''.*?'''",  # Docstrings
            r'#.*',         # Comments
            r'/\*.*?\*/',   # Block comments
            r'//.*',        # Line comments
            r'@\w+',        # Decorators/annotations
            r'->\s*\w+',    # Type hints
        ]
        
        docs_added = sum(
            len(re.findall(pattern, ' '.join(added_lines), re.DOTALL))
            for pattern in doc_patterns
        )
        
        docs_removed = sum(
            len(re.findall(pattern, ' '.join(removed_lines), re.DOTALL))
            for pattern in doc_patterns
        )
        
        doc_deficit = max(0, docs_removed - docs_added)
        
        score = min(1.0, doc_deficit / max(len(added_lines), 1) * 0.5)
        return score
    
    def _detect_assignment_issues(self, removed_lines, added_lines):
        """Detect assignment issues: wrong operators, precedence, side effects."""
        assignment_patterns = [
            r'[^=!<>]=(?!=)',  # Assignment (but not ==, !=, <=, >=)
            r'\+=', r'-=', r'\*=', r'/=', r'//=', r'%=',  # Compound assignment
            r'\+\+', r'--',  # Increment/decrement
        ]
        
        assign_count = sum(
            len(re.findall(pattern, ' '.join(added_lines + removed_lines)))
            for pattern in assignment_patterns
        )
        
        # Detect assignment in condition (risky pattern)
        cond_assign = len(re.findall(r'if\s*\(.*[^=!<>]=(?!=).*\)', ' '.join(added_lines)))
        
        score = min(1.0, (assign_count * 0.05 + cond_assign * 0.3) / max(len(added_lines), 1))
        return score
    
    def _detect_timing_issues(self, removed_lines, added_lines):
        """Detect timing/concurrency issues: locks, async, order dependencies."""
        timing_patterns = [
            r'lock\s*\(', r'Mutex', r'Semaphore', r'Monitor',
            r'async\s+', r'await\s+',
            r'\.start\(\)', r'\.join\(\)',
            r'thread', r'Thread', r'spawn',
            r'\.notify\(\)', r'\.wait\(\)',
        ]
        
        timing_ops = sum(
            len(re.findall(pattern, ' '.join(added_lines + removed_lines), re.IGNORECASE))
            for pattern in timing_patterns
        )
        
        # Detect ordering dependencies (multiple statements)
        ordering_risk = max(0, len(added_lines) - 1) if len(added_lines) > 3 else 0
        
        score = min(1.0, (timing_ops * 0.3 + ordering_risk * 0.01) / max(len(added_lines), 1))
        return score
    
    def _detect_build_issues(self, removed_lines, added_lines, filename):
        """Detect build/integration issues: imports, dependencies, configuration."""
        is_config = any(x in filename for x in ["setup.py", "package.json", "pom.xml", "Makefile", "CMakeLists"])
        is_import = any(x in filename for x in ["__init__", "init"])
        
        import_patterns = [
            r'import\s+', r'from\s+\w+\s+import',
            r'require\s*\(', r'load\s*\(',
            r'<include', r'<import',
        ]
        
        import_count = sum(
            len(re.findall(pattern, ' '.join(added_lines + removed_lines)))
            for pattern in import_patterns
        )
        
        version_patterns = [
            r'version\s*=', r'VERSION\s*=',
            r'"~', r'^\s*',  # Dependency version specifiers
        ]
        
        version_changes = sum(
            len(re.findall(pattern, ' '.join(added_lines + removed_lines)))
            for pattern in version_patterns
        )
        
        score = min(1.0, 
                   (import_count * 0.1 + version_changes * 0.2 + 
                    (2.0 if is_config else 0) + (0.5 if is_import else 0)) / max(len(added_lines), 1))
        return score
    
    def _compute_max_nesting(self, lines):
        """Estimate maximum nesting depth from indentation."""
        max_depth = 0
        for line in lines:
            if line.startswith(('-', '+')):
                line = line[1:]
            # Count leading whitespace as nesting level
            depth = len(line) - len(line.lstrip())
            depth = depth // 4 if depth >= 4 else depth // 2 if depth >= 2 else 0
            max_depth = max(max_depth, depth)
        return min(max_depth, 10)  # Cap at 10
    
    def compute_cognitive_load(self):
        """
        Phase 2: Compute Cognitive Load Theory (CLT) metrics.
        
        Three types of cognitive load:
        - Intrinsic: Inherent complexity of the problem domain
        - Extraneous: Unnecessary complexity in presentation/implementation
        - Germane: Effective working memory organization
        
        Total load = intrinsic + extraneous; germane contributes to learning/quality.
        
        **COMMIT-LEVEL GRANULARITY**: One metric per unique commit (sha).
        """
        logger.info("Computing cognitive load metrics (commit-level)...")
        
        # Use cached patterns directly to avoid large merge
        if self.patterns_df is None:
            logger.info("Loading cached patterns...")
            self.patterns_df = pd.read_parquet(DEFECT_PATTERNS_CACHE)
        
        # Ensure commit-level granularity (deduplicate on sha)
        patterns_by_sha = self.patterns_df.drop_duplicates(subset=['sha'], keep='first')
        logger.info("Deduplicating to commit-level: %d patterns → %d unique commits", 
                   len(self.patterns_df), len(patterns_by_sha))
        
        # Process in batches to avoid memory exhaustion
        batch_size = 10000
        cognitive_metrics = []
        
        for batch_start in range(0, len(patterns_by_sha), batch_size):
            batch_end = min(batch_start + batch_size, len(patterns_by_sha))
            batch = patterns_by_sha.iloc[batch_start:batch_end]
            
            for idx, (_, row) in enumerate(batch.iterrows()):
                if (batch_start + idx) % 5000 == 0:
                    logger.info("  [%d/%d] cognitive load computed...", batch_start + idx, len(patterns_by_sha))
                
                sha = row.get('sha')
                
                # Intrinsic load: problem complexity
                intrinsic = self._compute_intrinsic_load(row)
                
                # Extraneous load: poor code structure
                extraneous = self._compute_extraneous_load(row)
                
                # Germane load: effective organization
                germane = self._compute_germane_load(row, intrinsic)
                
                total_load = intrinsic + extraneous
                overload_risk = max(0, total_load - 7.0)  # Miller's magic number
                clt_category = self._categorize_clt_load(total_load)
                
                cognitive_metrics.append({
                    'sha': sha,
                    'intrinsic_load': intrinsic,
                    'extraneous_load': extraneous,
                    'germane_load': germane,
                    'total_cognitive_load': total_load,
                    'overload_risk': overload_risk,
                    'clt_category': clt_category,
                })
            
            # Clear batch from memory
            del batch
        
        self.cognitive_load_df = pd.DataFrame(cognitive_metrics)
        
        os.makedirs(os.path.dirname(COGNITIVE_LOAD_CACHE), exist_ok=True)
        self.cognitive_load_df.to_parquet(COGNITIVE_LOAD_CACHE, index=False)
        logger.info("Cognitive load computation complete: %d metrics cached (commit-level).", 
                   len(self.cognitive_load_df))
        
        return self.cognitive_load_df
    
    def _compute_intrinsic_load(self, row):
        """
        Intrinsic load = domain complexity + scope breadth.
        
        Indicators:
        - Multiple files changed (broader context needed)
        - Architectural files (core structure changes)
        - High cyclomatic complexity code
        - Async/concurrent patterns
        """
        base_load = 1.0
        
        # Complexity from PR size
        pr_size_loc = row.get('pr_size_loc', 0)
        if pr_size_loc > 500:
            base_load += 2.0
        elif pr_size_loc > 100:
            base_load += 1.0
        elif pr_size_loc > 20:
            base_load += 0.5
        
        # Complexity from nesting
        max_nesting = row.get('max_nesting_depth', 0)
        base_load += min(max_nesting / 5.0, 1.5)
        
        # Architectural complexity
        is_architecture = any(
            x in row.get('filename', '').lower() 
            for x in ['__init__', 'config', 'api', 'schema', 'interface', 'base', 'abstract']
        )
        if is_architecture:
            base_load += 1.0
        
        # Timing/async complexity
        timing_load = row.get('timing_issue_score', 0)
        base_load += timing_load * 1.5
        
        return min(base_load, 5.0)
    
    def _compute_extraneous_load(self, row):
        """
        Extraneous load = unnecessary complexity in code structure.
        
        Indicators:
        - Variable naming clarity
        - Code organization/structure
        - Duplication/refactoring debt
        - Documentation deficit
        """
        base_load = 0.0
        
        # Poor naming (high variable_count relative to lines)
        var_count = row.get('variable_count', 1)
        lines_changed = row.get('lines_changed', 1)
        naming_ratio = var_count / max(lines_changed, 1)
        if naming_ratio > 0.5:  # Many variables, dense code
            base_load += 1.0
        
        # Missing documentation
        doc_issue_score = row.get('documentation_issue_score', 0)
        base_load += doc_issue_score * 1.5
        
        # Code organization issues (nesting, assignment issues)
        assignment_score = row.get('assignment_error_score', 0)
        base_load += assignment_score * 1.0
        
        # Poor structure from data/interface issues
        data_score = row.get('data_error_score', 0)
        interface_score = row.get('interface_mismatch_score', 0)
        base_load += (data_score + interface_score) * 0.75
        
        return min(base_load, 4.0)
    
    def _compute_germane_load(self, row, intrinsic_load):
        """
        Germane load = how well solution organizes information.
        
        High germane load (> intrinsic) suggests good learning/understanding.
        Low germane load suggests solution doesn't effectively organize problem.
        """
        # Germane load is positive when author adds clarity/structure
        doc_gains = 1.0 - row.get('documentation_issue_score', 0)
        logic_clarity = 1.0 - row.get('logic_error_score', 0)
        checking_adds = 1.0 - row.get('missing_checks_score', 0)
        
        germane = (doc_gains + logic_clarity + checking_adds) / 3.0 * intrinsic_load
        
        return germane
    
    def _categorize_clt_load(self, total_load):
        """Categorize total cognitive load into buckets."""
        if total_load < 2.0:
            return 0  # LOW
        elif total_load < 4.0:
            return 1  # MODERATE
        elif total_load < 6.0:
            return 2  # HIGH
        else:
            return 3  # VERY_HIGH (overload risk)
    
    def detect_llm_patterns(self):
        """
        Phase 3: Detect LLM-specific error patterns based on probabilistic logic.
        
        LLM models generate text by predicting likely next tokens, which can lead to:
        - Off-by-one errors (common in loop/index generation)
        - Hallucinated APIs (predicting plausible-sounding but incorrect calls)
        - Inconsistent naming (renaming mid-function)
        - Missing error handling (low statistical weight in training)
        - Copy-paste errors (pattern repetition without adaptation)
        
        **COMMIT-LEVEL GRANULARITY**: One row per unique commit (sha), not per file.
        """
        logger.info("Detecting LLM-specific error patterns (commit-level)...")
        
        # Get UNIQUE commits (one per sha, deduplicate files within each commit)
        unique_commits = self.pr_commit_details_df[['sha', 'pr_id', 'patch']].drop_duplicates(subset=['sha'])
        
        llm_patterns_list = []
        for idx, (_, row) in enumerate(unique_commits.iterrows()):
            if idx % 5000 == 0:
                logger.info("  [%d/%d] LLM patterns detected...", idx, len(unique_commits))
            
            sha = row.get('sha')
            patch = str(row.get('patch', ''))
            
            if not patch or patch == 'nan':
                continue
            
            # Detect individual LLM error patterns
            off_by_one = self._detect_off_by_one_errors(patch)
            api_hallucination = self._detect_api_hallucination(patch)
            naming_inconsistency = self._detect_naming_inconsistency(patch)
            missing_error_handling = self._detect_missing_error_handling(patch)
            copy_paste_error = self._detect_copy_paste_errors(patch)
            
            # Composite LLM error likelihood (weighted)
            llm_likelihood = (
                off_by_one * 0.20 +
                api_hallucination * 0.30 +
                naming_inconsistency * 0.20 +
                missing_error_handling * 0.15 +
                copy_paste_error * 0.15
            )
            
            llm_patterns_list.append({
                'sha': sha,
                'llm_error_likelihood': llm_likelihood,
                'off_by_one_risk': off_by_one,
                'api_hallucination_risk': api_hallucination,
                'naming_inconsistency_risk': naming_inconsistency,
                'missing_error_handling_risk': missing_error_handling,
                'copy_paste_error_risk': copy_paste_error,
            })
        
        self.llm_patterns_df = pd.DataFrame(llm_patterns_list)
        
        os.makedirs(os.path.dirname(LLM_PATTERNS_CACHE), exist_ok=True)
        self.llm_patterns_df.to_parquet(LLM_PATTERNS_CACHE, index=False)
        logger.info("LLM pattern detection complete: %d commits analyzed (commit-level granularity).", 
                   len(self.llm_patterns_df))
        
        return self.llm_patterns_df
    
    def _detect_off_by_one_errors(self, patch):
        """Detect off-by-one errors: fence post errors, boundary issues."""
        # Patterns that often indicate off-by-one issues
        patterns = [
            r'i\s*<\s*len',  # < vs <=
            r'i\s*>\s*0',    # > vs >=
            r'\[.*\s*-\s*1\]',  # Array access with -1
            r'range\([^,]*,[^,]*,?\)',  # Range boundary
            r'for.*in\s+range',  # Loop boundary
        ]
        
        count = sum(len(re.findall(p, patch)) for p in patterns)
        
        # Look for boundaries that changed
        if '-' in patch and '+' in patch:
            removed_bounds = len(re.findall(r'<|>|<=|>=', patch.split('\n')[0]))
            added_bounds = len(re.findall(r'<|>|<=|>=', patch.split('\n')[-1]))
            count += abs(removed_bounds - added_bounds)
        
        score = min(1.0, count / 3.0)
        return score
    
    def _detect_api_hallucination(self, patch):
        """Detect hallucinated/incorrect API usage: wrong functions, parameters."""
        # API changes (function calls, method invocations)
        api_calls = len(re.findall(r'\w+\s*\(', patch))
        
        # Check for suspicious patterns (plausible but potentially wrong)
        suspicious = [
            r'\.\w+\(\)',  # Method calls without parameters
            r'new\s+\w+\(',  # Constructor calls
            r'import\s+\w+',  # Import statements
        ]
        
        suspicious_count = sum(len(re.findall(p, patch)) for p in suspicious)
        
        # If many API calls changed, higher hallucination risk
        score = min(1.0, api_calls * 0.1 + suspicious_count * 0.15)
        return score
    
    def _detect_naming_inconsistency(self, patch):
        """Detect naming inconsistencies: renaming without full refactoring."""
        # Extract variable names from removed and added lines
        removed_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\b', 
                                      '\n'.join([l for l in patch.split('\n') if l.startswith('-')])))
        added_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\b',
                                    '\n'.join([l for l in patch.split('\n') if l.startswith('+')])))
        
        # High inconsistency if variable sets don't overlap much
        if removed_vars and added_vars:
            overlap = len(removed_vars & added_vars) / len(removed_vars | added_vars)
            inconsistency = 1.0 - overlap
        else:
            inconsistency = 0.0
        
        # Also check for mid-function renaming
        lines = patch.split('\n')
        renaming_instances = sum(
            1 for i in range(1, len(lines))
            if lines[i].startswith('+') and 'rename' in lines[i].lower()
        )
        
        score = min(1.0, inconsistency * 0.7 + renaming_instances * 0.1)
        return score
    
    def _detect_missing_error_handling(self, patch):
        """Detect missing error handling: try/catch, null checks absent."""
        # Count error handling patterns
        try_catches = len(re.findall(r'(try|except|catch|finally|throw|raise)', patch, re.IGNORECASE))
        
        # Patterns that should have error handling
        risky_patterns = [
            r'\.parse\(',
            r'\.open\(',
            r'\.request\(',
            r'\.read\(',
            r'\.load\(',
        ]
        
        risky_count = sum(len(re.findall(p, patch)) for p in risky_patterns)
        
        # If risky operations without error handling, high score
        if risky_count > 0 and try_catches == 0:
            score = min(1.0, risky_count * 0.3)
        else:
            score = max(0.0, (risky_count - try_catches) * 0.1)
        
        return score
    
    def _detect_copy_paste_errors(self, patch):
        """Detect copy-paste errors: repeated code without adaptation."""
        lines = patch.split('\n')
        added_lines = [l[1:] for l in lines if l.startswith('+') and not l.startswith('+++')]
        
        # Find similar lines (indicating copy-paste)
        if len(added_lines) < 2:
            return 0.0
        
        similarity_scores = []
        for i in range(len(added_lines) - 1):
            for j in range(i + 1, min(i + 5, len(added_lines))):
                # Simple similarity: character overlap
                s1, s2 = added_lines[i], added_lines[j]
                common = sum(1 for c in s1 if c in s2)
                max_len = max(len(s1), len(s2))
                if max_len > 0:
                    similarity = common / max_len
                    if similarity > 0.7:  # High similarity
                        similarity_scores.append(similarity)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            score = min(1.0, avg_similarity * 0.5)
        else:
            score = 0.0
        
        return score
    
    def classify_to_odc(self):
        """
        Phase 4: Assign ODC classification based on patterns and context.
        
        Logic:
        1. Find dominant defect type from pattern scores
        2. Determine structural class from code context
        3. Infer trigger context from PR metadata
        4. Estimate impact from defect severity and spread
        
        **COMMIT-LEVEL GRANULARITY**: One row per unique commit (sha), not per file.
        Aggregates file-level data to commit level.
        """
        logger.info("Classifying defects to ODC taxonomy (commit-level)...")
        
        # Load cached data separately (avoid large merge)
        patterns_df = self.patterns_df if self.patterns_df is not None else pd.read_parquet(DEFECT_PATTERNS_CACHE)
        clt_df = self.cognitive_load_df if self.cognitive_load_df is not None else pd.read_parquet(COGNITIVE_LOAD_CACHE)
        llm_df = self.llm_patterns_df if self.llm_patterns_df is not None else pd.read_parquet(LLM_PATTERNS_CACHE)
        
        # CRITICAL: Get UNIQUE commits (one per sha, not per file)
        # pr_commit_details_df has one row per file per commit; we need to deduplicate on sha
        unique_commits = self.pr_commit_details_df[['pr_id', 'sha']].drop_duplicates(subset=['sha'])
        pr_to_shas = {pr_id: group['sha'].unique().tolist() 
                      for pr_id, group in unique_commits.groupby('pr_id')}
        
        # Create indexed lookup dictionaries to avoid large merges
        patterns_dict = {sha: row.to_dict() for sha, row in patterns_df.set_index('sha').iterrows()}
        clt_dict = {sha: row.to_dict() for sha, row in clt_df.set_index('sha').iterrows()}
        llm_dict = {sha: row.to_dict() for sha, row in llm_df.set_index('sha').iterrows()}
        
        # Create PR metadata lookup (id -> PR row)
        pr_dict = {row.get('id'): row.to_dict() for _, row in self.main_frame.iterrows()}
        
        classifications = []
        total_shas = sum(len(shas) for shas in pr_to_shas.values())
        shas_processed = 0
        
        # Iterate through PRs and their unique commits
        for pr_id, shas in pr_to_shas.items():
            if pr_id not in pr_dict:
                continue
            
            pr_row = pr_dict[pr_id]
            
            # Iterate over UNIQUE commits for this PR
            for sha in shas:
                shas_processed += 1
                if shas_processed % 5000 == 0:
                    logger.info("  [%d/%d] commits classified...", shas_processed, total_shas)
                
                # Skip if no pattern data for this commit
                if sha not in patterns_dict:
                    continue
                
                pattern_row = patterns_dict[sha]
                clt_row = clt_dict.get(sha, {})
                llm_row = llm_dict.get(sha, {})
                
                # Determine ODC type (find max scoring pattern)
                type_scores = {
                    0: pattern_row.get('api_misuse_score', 0),
                    1: pattern_row.get('logic_error_score', 0),
                    2: pattern_row.get('interface_mismatch_score', 0),
                    3: pattern_row.get('data_error_score', 0),
                    4: pattern_row.get('missing_checks_score', 0),
                    5: pattern_row.get('documentation_issue_score', 0),
                    6: pattern_row.get('assignment_error_score', 0),
                    7: pattern_row.get('timing_issue_score', 0),
                    8: pattern_row.get('build_error_score', 0),
                }
                odc_type = max(type_scores, key=type_scores.get)
                
                # Determine ODC class (structural aspect)
                odc_class = self._assign_odc_class_simple('', pr_row.get('pr_size_loc', 0), odc_type)
                
                # Determine ODC trigger (discovery context)
                odc_trigger = self._assign_odc_trigger_simple(pr_row)
                
                # Determine ODC impact (severity)
                odc_impact = self._assign_odc_impact_simple(type_scores)
                
                classifications.append({
                    'id': pr_id,
                    'sha': sha,
                    'odc_type': odc_type,
                    'odc_type_name': ODC_TYPE_GROUPS[odc_type],
                    'odc_class': odc_class,
                    'odc_class_name': ODC_CLASS_GROUPS[odc_class],
                    'odc_trigger': odc_trigger,
                    'odc_trigger_name': ODC_TRIGGER_GROUPS[odc_trigger],
                    'odc_impact': odc_impact,
                    'odc_impact_name': ODC_IMPACT_GROUPS[odc_impact],
                })
        
        self.odc_df = pd.DataFrame(classifications)
        logger.info("ODC classification complete: %d commits classified (commit-level granularity).", 
                   len(self.odc_df))
        
        # Cache ODC classifications for later retrieval
        odc_cache = os.path.join(BASE_DIR, "data", "processed", "rq3_odc_classifications.parquet")
        os.makedirs(os.path.dirname(odc_cache), exist_ok=True)
        self.odc_df.to_parquet(odc_cache, index=False)
        logger.info("ODC classifications cached to %s", odc_cache)
        
        return self.odc_df
    
    def _assign_odc_class(self, row, odc_type):
        """Assign structural class based on code context."""
        filename = str(row.get('filename', '')).lower()
        files_changed = row.get('lines_changed', 0)
        
        # Missing feature or docs
        if 'readme' in filename or '.md' in filename:
            return 5  # DOCUMENTATION
        
        # Build/integration issues span multiple files
        if files_changed > 50:
            return 3  # BUILD_INTEGRATION
        
        # Interface issues suggest multi-module interaction
        if odc_type == 2:  # Interface
            return 1  # INTERFACE
        
        # Timing/concurrency usually within module
        if odc_type == 7:  # Timing
            return 0  # FUNCTION (usually)
        
        # Default to function-level for now
        return 0  # FUNCTION
    
    def _assign_odc_class_simple(self, filename, files_changed, odc_type):
        """Simplified version for use in classify_to_odc with indexed lookups."""
        # Missing feature or docs
        if 'readme' in filename or '.md' in filename:
            return 5  # DOCUMENTATION
        
        # Build/integration issues span multiple files
        if files_changed > 50:
            return 3  # BUILD_INTEGRATION
        
        # Interface issues suggest multi-module interaction
        if odc_type == 2:  # Interface
            return 1  # INTERFACE
        
        # Timing/concurrency usually within module
        if odc_type == 7:  # Timing
            return 0  # FUNCTION (usually)
        
        # Default to function-level
        return 0  # FUNCTION
    
    def _assign_odc_trigger(self, row):
        """Infer discovery trigger from PR context."""
        filename = str(row.get('filename', '')).lower()
        task_type = str(row.get('task_type', '')).lower()
        
        # Test-related files suggest unit test discovery
        if any(x in filename for x in ['test', 'spec', '__test__']):
            return 1  # UNIT_TEST
        
        # Fix PRs suggest field/production discovery
        if 'fix' in task_type:
            return 4  # FIELD_TEST (fix for production issue)
        
        # Docs suggest code review
        if 'doc' in task_type or 'README' in row.get('filename', ''):
            return 0  # CODE_REVIEW
        
        # Default to code review
        return 0  # CODE_REVIEW
    
    def _assign_odc_trigger_simple(self, row):
        """Simplified version for use in classify_to_odc with indexed lookups."""
        task_type_group = str(row.get('task_type_group', '')).lower()
        
        # Fix PRs suggest field/production discovery
        if 'fix' in task_type_group:
            return 4  # FIELD_TEST
        
        # Default to code review
        return 0  # CODE_REVIEW
    
    def _assign_odc_impact(self, row, type_scores):
        """Estimate impact from defect type and severity."""
        max_score = max(type_scores.values())
        max_type = max(type_scores, key=type_scores.get)
        
        # Critical types → higher impact
        critical_types = {7, 3, 4}  # Timing, Data, Checking
        
        if max_type in critical_types and max_score > 0.7:
            return 0  # CRITICAL
        elif max_score > 0.6:
            return 1  # MAJOR
        elif max_score > 0.4:
            return 2  # MODERATE
        else:
            return 3  # MINOR
    
    def _assign_odc_impact_simple(self, type_scores):
        """Simplified version for use in classify_to_odc."""
        max_score = max(type_scores.values())
        max_type = max(type_scores, key=type_scores.get)
        
        # Critical types → higher impact
        critical_types = {7, 3, 4}  # Timing, Data, Checking
        
        if max_type in critical_types and max_score > 0.7:
            return 0  # CRITICAL
        elif max_score > 0.6:
            return 1  # MAJOR
        elif max_score > 0.4:
            return 2  # MODERATE
        else:
            return 3  # MINOR
    
    def build_final_frame(self):
        """
        Combine all analyses into final output frame (commit-level).
        
        Structure: One row per commit (sha), with PR-level metadata joined.
        No Cartesian products or duplication.
        """
        logger.info("Building final defect classifier frame (commit-level)...")
        
        # Load ODC classifications if not in memory (e.g., when running --stage 5 directly)
        odc_df = self.odc_df
        if odc_df is None or odc_df.empty:
            # Try to load from intermediate cache (if it exists)
            odc_cache = os.path.join(BASE_DIR, "data", "processed", "rq3_odc_classifications.parquet")
            if os.path.exists(odc_cache):
                logger.info("Loading cached ODC classifications...")
                odc_df = pd.read_parquet(odc_cache)
            else:
                logger.error("No ODC classifications found. Cannot build final frame.")
                return None
        
        # Load main_frame if not in memory
        if self.main_frame is None:
            logger.info("Loading RQ1 main frame...")
            self.main_frame = pd.read_parquet(RQ1_MAIN_FRAME)
        
        # Start with odc_df (commit-level: one row per sha)
        result = odc_df.copy()
        logger.info("Starting with %d commits from ODC classifications", len(result))
        
        # Load cached signal data if not in memory
        clt_df = self.cognitive_load_df if self.cognitive_load_df is not None else None
        llm_df = self.llm_patterns_df if self.llm_patterns_df is not None else None
        
        if clt_df is None and os.path.exists(COGNITIVE_LOAD_CACHE):
            clt_df = pd.read_parquet(COGNITIVE_LOAD_CACHE)
        if llm_df is None and os.path.exists(LLM_PATTERNS_CACHE):
            llm_df = pd.read_parquet(LLM_PATTERNS_CACHE)
        
        # Create lookup dictionaries for sha → signals (avoid merge)
        clt_dict = {}
        if clt_df is not None and not clt_df.empty:
            for sha, row in clt_df.drop_duplicates(subset=['sha']).set_index('sha').iterrows():
                clt_dict[sha] = row.to_dict()
        
        llm_dict = {}
        if llm_df is not None and not llm_df.empty:
            for sha, row in llm_df.drop_duplicates(subset=['sha']).set_index('sha').iterrows():
                llm_dict[sha] = row.to_dict()
        
        # Add CLT columns via lookup (no merge)
        if clt_dict:
            clt_sample = next(iter(clt_dict.values()))
            clt_cols = [c for c in clt_sample.keys() if c != 'sha' and c != 'pr_id']
            for col in clt_cols:
                result[col] = result['sha'].map(lambda x: clt_dict.get(x, {}).get(col, np.nan))
            logger.info("Added %d CLT columns", len(clt_cols))
        
        # Add LLM columns via lookup (no merge)
        if llm_dict:
            llm_sample = next(iter(llm_dict.values()))
            llm_cols = [c for c in llm_sample.keys() if c != 'sha' and c != 'pr_id']
            for col in llm_cols:
                result[col] = result['sha'].map(lambda x: llm_dict.get(x, {}).get(col, np.nan))
            logger.info("Added %d LLM columns", len(llm_cols))
        
        # Finally, merge PR-level metadata from main_frame (one row per PR)
        # Use left merge to preserve all commits, join PR metadata
        pr_metadata = self.main_frame[['id', 'repo_full_name', 'domain_type_group', 
                                        'task_type_group', 'created_at', 'ai_pr', 'number', 'user',
                                        'pr_size_loc']].copy()
        
        result = result.merge(
            pr_metadata,
            left_on='id',
            right_on='id',
            how='left',
            suffixes=('_commit', '_pr')
        )
        
        logger.info("Final frame after PR metadata merge: %d rows (commit-level)", len(result))
        
        # Select columns in desired order
        column_order = [
            # Identifiers
            'id', 'sha', 'ai_pr', 'number', 'user',
            
            # PR metadata
            'repo_full_name', 'domain_type_group', 'task_type_group',
            'created_at', 'pr_size_loc',
            
            # Cognitive Load Theory signals
            'intrinsic_load', 'extraneous_load', 'germane_load', 'total_cognitive_load', 'clt_category', 'overload_risk',
            
            # LLM error signature signals
            'llm_error_likelihood',
            'off_by_one_risk', 'api_hallucination_risk', 'naming_inconsistency_risk',
            'missing_error_handling_risk', 'copy_paste_error_risk',
            
            # ODC classification (main results)
            'odc_type', 'odc_type_name',
            'odc_class', 'odc_class_name',
            'odc_trigger', 'odc_trigger_name',
            'odc_impact', 'odc_impact_name',
        ]
        
        # Keep only columns that exist
        column_order = [c for c in column_order if c in result.columns]
        final = result[column_order].copy()
        
        # Save outputs
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        final.to_parquet(OUTPUT_PATH, index=False)
        final.to_csv(OUTPUT_CSV_PATH, index=False)
        logger.info("Final frame saved: %d rows (commit-level) → %s", len(final), OUTPUT_PATH)
        
        self.final_frame = final
        return final
    
    def save_checkpoint(self):
        """Save current progress to checkpoint."""
        if self.main_frame is not None and not self.main_frame.empty:
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            self.main_frame.to_parquet(CHECKPOINT_PATH, index=False)
            logger.info("Checkpoint saved: %d rows", len(self.main_frame))


def main():
    parser = argparse.ArgumentParser(description="RQ3: Orthogonal Defect Classification (ODC) Ingest")
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='Force rebuild all stages, ignore cached files')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--stage', type=int, default=0, 
                       help='Stage to start from: 0=load, 1=patterns, 2=clt, 3=llm, 4=odc, 5=final')
    args = parser.parse_args()
    
    classifier = DefectClassifier()
    
    # Load data
    if args.stage <= 0:
        classifier.load_rq1_data()
        logger.info("Data loading complete.")
    
    # Stage 1: Extract code patterns (with caching)
    if args.stage <= 1:
        if not args.force_rebuild and os.path.exists(DEFECT_PATTERNS_CACHE):
            logger.info("Loading cached patterns from %s", DEFECT_PATTERNS_CACHE)
            classifier.patterns_df = pd.read_parquet(DEFECT_PATTERNS_CACHE)
            logger.info("Pattern extraction skipped (cached: %d patterns)", len(classifier.patterns_df))
        else:
            classifier.extract_code_patterns()
            logger.info("Pattern extraction complete.")
    
    # Stage 2: Compute cognitive load (with caching)
    if args.stage <= 2:
        if not args.force_rebuild and os.path.exists(COGNITIVE_LOAD_CACHE):
            logger.info("Loading cached cognitive load from %s", COGNITIVE_LOAD_CACHE)
            classifier.cognitive_load_df = pd.read_parquet(COGNITIVE_LOAD_CACHE)
            logger.info("Cognitive load computation skipped (cached: %d metrics)", len(classifier.cognitive_load_df))
        else:
            classifier.compute_cognitive_load()
            logger.info("Cognitive load computation complete.")
    
    # Stage 3: Detect LLM patterns (with caching)
    if args.stage <= 3:
        if not args.force_rebuild and os.path.exists(LLM_PATTERNS_CACHE):
            logger.info("Loading cached LLM patterns from %s", LLM_PATTERNS_CACHE)
            classifier.llm_patterns_df = pd.read_parquet(LLM_PATTERNS_CACHE)
            logger.info("LLM pattern detection skipped (cached: %d patterns)", len(classifier.llm_patterns_df))
        else:
            classifier.detect_llm_patterns()
            logger.info("LLM pattern detection complete.")
    
    # Stage 4: Classify to ODC
    if args.stage <= 4:
        classifier.classify_to_odc()
        logger.info("ODC classification complete.")
    
    # Stage 5: Build final frame
    if args.stage <= 5:
        classifier.build_final_frame()
        logger.info("Final frame built.")
    
    logger.info("=" * 60)
    logger.info("RQ3 PIPELINE COMPLETE")
    if classifier.main_frame is not None:
        logger.info("Total classifications: %d", len(classifier.main_frame))
        logger.info("Columns: %s", list(classifier.main_frame.columns))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


# ============================================================================
# KEY DIFFERENCES FROM RQ1_INGEST.PY
# ============================================================================
#
# 1. **ANALYSIS UNIT**
#    RQ1: Pull requests (coarse-grained, system-level outcomes)
#    RQ3: Individual commits (fine-grained, defect type classification)
#    → Data flows from RQ1 output to commit-level details in RQ3
#
# 2. **DEFECT DETECTION**
#    RQ1: Heuristic-based (file overlap, PR references, issue linkage)
#    RQ3: Pattern-based static code analysis (API calls, logic, checks, etc.)
#    → RQ3 extracts specific code patterns from patches
#
# 3. **CLASSIFICATION SCHEMA**
#    RQ1: Binary outcome (has_defect_90d) + quantitative metrics
#    RQ3: Multidimensional ODC taxonomy (type × class × trigger × impact)
#    → RQ3 assigns 4 orthogonal dimensions per defect
#
# 4. **EXPLANATORY VARIABLES**
#    RQ1: Repository/contributor/task metadata (focuses on context)
#    RQ3: Code patterns + cognitive load + LLM signatures (focuses on code)
#    → RQ3 adds 9 pattern scores, 6 cognitive load metrics, 5 LLM indicators
#
# 5. **THEORETICAL FRAMEWORK**
#    RQ1: Empirical comparison (AI vs Human maintenance burden)
#    RQ3: Theoretical grounding (Cognitive Load Theory + LLM probabilistics)
#    → RQ3 measures constructs (intrinsic/extraneous/germane load)
#
# 6. **CACHING STRATEGY**
#    RQ1: 4 caches (human PR stats, commits, reviews, files)
#    RQ3: 3 caches (defect patterns, cognitive load, LLM patterns)
#    → RQ3 caches intermediate analyses to enable selective recomputation
#
# 7. **ANALYSIS UNIT RELATIONSHIP**
#    RQ1: PR-level aggregation (commits → PR outcomes)
#    RQ3: Commit-level granularity (raw patches → fine-grained classification)
#    → RQ3 can trace defects back to specific commits via sha
#
# 8. **DATA VOLUME**
#    RQ1: ~500K PRs from popular repos (GitHub API limited)
#    RQ3: ~millions of commits (from parquet, no API limit)
#    → RQ3 can work at much finer granularity due to availability
#
# 9. **MISSING DATA STRATEGY**
#    RQ1: Drops PRs with incomplete data, uses caching to manage API calls
#    RQ3: Works with patch diffs (may be incomplete but more available)
#    → RQ3 gracefully handles missing patches via heuristic fallbacks
#
# 10. **EXTENSIBILITY**
#     RQ1: Designed for 90-day post-merge outcomes
#     RQ3: Framework supports future integration of:
#          - AST-based static analysis (radon, libcst)
#          - ML-based defect prediction (scikit-learn)
#          - Tool-based analysis (Pylint, ESLint, SonarQube)
#     → RQ3 has pluggable pattern detection
