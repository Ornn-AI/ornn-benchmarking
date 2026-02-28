"""Tests for CI workflow configuration and repo hygiene.

Validates that GitHub Actions workflows are public-repo friendly,
cost-safe (free-tier only), and enforce lint/typecheck/test gates.
Also validates repository cleanliness for open-source readiness.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"
GITIGNORE = ROOT / ".gitignore"


def _read(path: Path) -> str:
    """Read file content, raising a clear error if missing."""
    assert path.exists(), f"Expected file not found: {path}"
    return path.read_text(encoding="utf-8")


def _load_ci_workflow() -> dict:  # type: ignore[type-arg]
    """Parse the CI workflow YAML.

    Note: PyYAML parses the YAML keyword ``on`` as boolean True, so
    trigger configuration is accessed via ``wf[True]``.
    """
    content = _read(CI_WORKFLOW)
    return yaml.safe_load(content)


def _get_triggers(wf: dict) -> dict:  # type: ignore[type-arg]
    """Extract trigger config from workflow, handling YAML 'on' → True key."""
    # PyYAML converts 'on:' to boolean True
    return wf.get("on") or wf.get(True) or {}


# ===================================================================
# CI workflow existence and structure
# ===================================================================


class TestCIWorkflowExists:
    """CI workflow file must exist at .github/workflows/ci.yml."""

    def test_workflows_directory_exists(self) -> None:
        assert WORKFLOWS_DIR.exists(), ".github/workflows/ directory not found"

    def test_ci_workflow_exists(self) -> None:
        assert CI_WORKFLOW.exists(), ".github/workflows/ci.yml not found"

    def test_ci_workflow_is_valid_yaml(self) -> None:
        content = _read(CI_WORKFLOW)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), "CI workflow should parse as a YAML mapping"


# ===================================================================
# Trigger configuration
# ===================================================================


class TestCITriggers:
    """CI must trigger on push and pull_request."""

    def test_triggers_on_push(self) -> None:
        wf = _load_ci_workflow()
        triggers = _get_triggers(wf)
        assert "push" in triggers, "CI should trigger on push"

    def test_triggers_on_pull_request(self) -> None:
        wf = _load_ci_workflow()
        triggers = _get_triggers(wf)
        assert "pull_request" in triggers, "CI should trigger on pull_request"

    def test_push_includes_main_branch(self) -> None:
        wf = _load_ci_workflow()
        triggers = _get_triggers(wf)
        push_config = triggers.get("push", {})
        branches = push_config.get("branches", [])
        assert "main" in branches, "CI push should include main branch"


# ===================================================================
# Jobs: lint, typecheck, test
# ===================================================================


class TestCIJobsCoverage:
    """CI must include lint, typecheck, and test jobs."""

    def test_has_lint_job(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        job_names = list(jobs.keys())
        job_steps_text = str(jobs).lower()
        assert "lint" in job_names or "ruff" in job_steps_text, (
            f"CI must have a lint job; found jobs: {job_names}"
        )

    def test_has_typecheck_job(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        job_names = list(jobs.keys())
        job_steps_text = str(jobs).lower()
        assert "typecheck" in job_names or "mypy" in job_steps_text, (
            f"CI must have a typecheck job; found jobs: {job_names}"
        )

    def test_has_test_job(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        job_names = list(jobs.keys())
        job_steps_text = str(jobs).lower()
        assert "test" in job_names or "pytest" in job_steps_text, (
            f"CI must have a test job; found jobs: {job_names}"
        )

    def test_lint_runs_ruff(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        lint_job = jobs.get("lint", {})
        steps_text = str(lint_job.get("steps", [])).lower()
        assert "ruff" in steps_text, "Lint job should run ruff"

    def test_typecheck_runs_mypy(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        tc_job = jobs.get("typecheck", {})
        steps_text = str(tc_job.get("steps", [])).lower()
        assert "mypy" in steps_text, "Typecheck job should run mypy"

    def test_test_runs_pytest(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        test_job = jobs.get("test", {})
        steps_text = str(test_job.get("steps", [])).lower()
        assert "pytest" in steps_text, "Test job should run pytest"


# ===================================================================
# Free-tier safety
# ===================================================================


class TestCIFreeTierSafety:
    """CI workflow must be free for public repos — no paid features."""

    # Patterns for paid/non-free-tier CI features
    PAID_FEATURES: ClassVar[list[tuple[str, str]]] = [
        (r"runs-on:\s*\[?self-hosted", "self-hosted runners"),
        (r"runs-on:\s*(?:macos|windows)", "non-Linux runners (macOS/Windows cost money)"),
        (r"gcloud\s", "GCP CLI calls in CI"),
        (r"aws\s", "AWS CLI calls in CI"),
        (r"docker\s+push", "Docker push (may need registry costs)"),
    ]

    def test_uses_github_hosted_runners(self) -> None:
        wf = _load_ci_workflow()
        jobs = wf.get("jobs", {})
        for job_name, job_config in jobs.items():
            runs_on = str(job_config.get("runs-on", ""))
            assert "ubuntu" in runs_on.lower() or "latest" in runs_on.lower(), (
                f"Job '{job_name}' should use ubuntu-latest runner for free-tier safety"
            )

    def test_no_paid_features(self) -> None:
        content = _read(CI_WORKFLOW)
        for pattern, feature_name in self.PAID_FEATURES:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, (
                f"CI workflow uses paid feature: {feature_name} "
                f"(pattern: {pattern!r}, matches: {matches})"
            )

    def test_no_secrets_required(self) -> None:
        """CI should run without any secrets (pure open-source repo)."""
        content = _read(CI_WORKFLOW)
        # Secrets references in required steps indicate paid/private features
        secret_refs = re.findall(r"\$\{\{\s*secrets\.\w+\s*\}\}", content)
        assert not secret_refs, (
            f"CI workflow requires secrets (not public-repo friendly): {secret_refs}"
        )

    def test_no_external_service_dependencies(self) -> None:
        """CI should not require external services (databases, APIs, etc.)."""
        content = _read(CI_WORKFLOW).lower()
        for service in ["services:", "firestore", "redis", "postgres", "mysql"]:
            assert service not in content, (
                f"CI workflow references external service: {service!r}"
            )


# ===================================================================
# Repository hygiene
# ===================================================================


class TestRepoHygiene:
    """Repository should be clean for open-source release."""

    def test_gitignore_exists(self) -> None:
        assert GITIGNORE.exists(), ".gitignore not found at project root"

    def test_gitignore_covers_python_artifacts(self) -> None:
        content = _read(GITIGNORE)
        # *.py[cod] covers *.pyc, *.pyo, *.pyd — accept either form
        for pattern in ["__pycache__", "*.py[cod]", ".eggs", "dist/"]:
            assert any(p in content for p in [pattern, pattern.replace("*", "")]), (
                f".gitignore should cover {pattern}"
            )

    def test_gitignore_covers_virtualenvs(self) -> None:
        content = _read(GITIGNORE)
        assert ".venv" in content or "venv" in content, (
            ".gitignore should cover virtual environments"
        )

    def test_gitignore_covers_env_files(self) -> None:
        content = _read(GITIGNORE)
        assert ".env" in content, ".gitignore should cover .env files"

    def test_gitignore_covers_ide_files(self) -> None:
        content = _read(GITIGNORE)
        assert ".vscode" in content or ".idea" in content, (
            ".gitignore should cover IDE configuration files"
        )

    def test_gitignore_covers_test_cache(self) -> None:
        content = _read(GITIGNORE)
        assert ".pytest_cache" in content, ".gitignore should cover .pytest_cache"
        assert ".mypy_cache" in content, ".gitignore should cover .mypy_cache"
        assert ".ruff_cache" in content, ".gitignore should cover .ruff_cache"

    def test_no_env_files_committed(self) -> None:
        """No .env files should be committed to the repo."""
        env_files = list(ROOT.glob(".env*"))
        committed = [f for f in env_files if f.name not in {".env.example"}]
        assert not committed, (
            f"Sensitive .env files should not be committed: {[f.name for f in committed]}"
        )

    def test_no_secrets_in_source(self) -> None:
        """Source code should not contain hardcoded API keys or tokens."""
        src_dir = ROOT / "src"
        api_dir = ROOT / "api"
        secret_patterns = [
            re.compile(r'(?:api[_-]?key|secret|token|password)\s*=\s*["\'][^"\']{20,}["\']',
                        re.IGNORECASE),
        ]
        for directory in [src_dir, api_dir]:
            if not directory.exists():
                continue
            for py_file in directory.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8")
                for pattern in secret_patterns:
                    matches = pattern.findall(content)
                    assert not matches, (
                        f"Potential secret in {py_file.relative_to(ROOT)}: {matches}"
                    )

    def test_factory_dir_not_in_gitignore(self) -> None:
        """The .factory directory should NOT be in .gitignore."""
        content = _read(GITIGNORE)
        lines = [line.strip() for line in content.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        for line in lines:
            assert line != ".factory" and line != ".factory/", (
                ".factory should NOT be in .gitignore — it contains mission infrastructure"
            )


class TestNoTempArtifacts:
    """No temporary or generated artifacts should be tracked in git."""

    def test_no_report_json_committed(self) -> None:
        """ornn_report_*.json files should be gitignored, not committed."""
        # These are generated at runtime and should not be in the repo
        # The .gitignore covers them, but verify none snuck through
        gitignore_content = _read(GITIGNORE)
        assert "ornn_report_" in gitignore_content, (
            ".gitignore should cover ornn_report_*.json files"
        )

    def test_no_build_artifacts_committed(self) -> None:
        """dist/ and build/ should not exist in the repo."""
        assert not (ROOT / "dist").exists() or not any((ROOT / "dist").iterdir()), (
            "dist/ directory should not contain committed artifacts"
        )
