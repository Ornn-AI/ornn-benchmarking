"""Tests for PyPI publish workflow configuration.

Validates that the GitHub Actions publish workflow uses Trusted Publisher
(OIDC), triggers on push to main, and remains free-tier-safe.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
PUBLISH_WORKFLOW = WORKFLOWS_DIR / "publish.yml"


def _read(path: Path) -> str:
    """Read file content, raising a clear error if missing."""
    assert path.exists(), f"Expected file not found: {path}"
    return path.read_text(encoding="utf-8")


def _load_publish_workflow() -> dict:  # type: ignore[type-arg]
    """Parse the publish workflow YAML."""
    content = _read(PUBLISH_WORKFLOW)
    return yaml.safe_load(content)


def _get_triggers(wf: dict) -> dict:  # type: ignore[type-arg]
    """Extract trigger config, handling YAML 'on' → True key."""
    return wf.get("on") or wf.get(True) or {}


# ===================================================================
# Workflow existence and validity
# ===================================================================


class TestPublishWorkflowExists:
    """Publish workflow must exist at .github/workflows/publish.yml."""

    def test_publish_workflow_exists(self) -> None:
        assert PUBLISH_WORKFLOW.exists(), ".github/workflows/publish.yml not found"

    def test_publish_workflow_is_valid_yaml(self) -> None:
        content = _read(PUBLISH_WORKFLOW)
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict), "Publish workflow should parse as a YAML mapping"


# ===================================================================
# Trigger configuration
# ===================================================================


class TestPublishTriggers:
    """Publish workflow must trigger on push to main."""

    def test_triggers_on_push(self) -> None:
        wf = _load_publish_workflow()
        triggers = _get_triggers(wf)
        assert "push" in triggers, "Publish workflow should trigger on push"

    def test_push_includes_main_branch(self) -> None:
        wf = _load_publish_workflow()
        triggers = _get_triggers(wf)
        push_config = triggers.get("push", {})
        branches = push_config.get("branches", [])
        assert "main" in branches, "Publish workflow push should include main branch"


# ===================================================================
# Trusted Publisher (OIDC) configuration
# ===================================================================


class TestTrustedPublisher:
    """Publish workflow must use PyPI Trusted Publisher (no API token)."""

    def test_has_id_token_write_permission(self) -> None:
        """Trusted Publishing requires id-token: write permission."""
        wf = _load_publish_workflow()
        jobs = wf.get("jobs", {})
        # Check for id-token: write at job level (preferred) or workflow level
        workflow_perms = wf.get("permissions", {})
        has_workflow_level = (
            isinstance(workflow_perms, dict) and workflow_perms.get("id-token") == "write"
        )
        has_job_level = False
        for job_config in jobs.values():
            perms = job_config.get("permissions", {})
            if isinstance(perms, dict) and perms.get("id-token") == "write":
                has_job_level = True
                break
        assert has_workflow_level or has_job_level, (
            "Publish workflow must have 'id-token: write' permission for Trusted Publishing"
        )

    def test_uses_pypa_publish_action(self) -> None:
        """Workflow must use pypa/gh-action-pypi-publish."""
        content = _read(PUBLISH_WORKFLOW)
        assert "pypa/gh-action-pypi-publish" in content, (
            "Publish workflow must use pypa/gh-action-pypi-publish action"
        )

    def test_no_api_token_or_password(self) -> None:
        """Trusted Publisher should not use API tokens or passwords."""
        content = _read(PUBLISH_WORKFLOW)
        # Should not reference secrets for PyPI auth
        assert "PYPI_TOKEN" not in content, (
            "Trusted Publisher workflow should not use PYPI_TOKEN secret"
        )
        assert "PYPI_API_TOKEN" not in content, (
            "Trusted Publisher workflow should not use PYPI_API_TOKEN secret"
        )
        # Should not use username/password pattern
        secret_refs = re.findall(r"\$\{\{\s*secrets\.\w+\s*\}\}", content)
        assert not secret_refs, (
            f"Trusted Publisher workflow should not require secrets: {secret_refs}"
        )


# ===================================================================
# Build and publish job structure
# ===================================================================


class TestPublishJobStructure:
    """Publish workflow must build and publish correctly."""

    def test_has_build_step(self) -> None:
        """Workflow must build the distribution."""
        content = _read(PUBLISH_WORKFLOW).lower()
        assert "python -m build" in content, (
            "Publish workflow must include 'python -m build' step"
        )

    def test_has_publish_job(self) -> None:
        """Workflow must have a job that publishes to PyPI."""
        wf = _load_publish_workflow()
        jobs = wf.get("jobs", {})
        job_text = str(jobs).lower()
        assert "pypi" in job_text or "publish" in job_text, (
            "Publish workflow must have a publish/pypi job"
        )

    def test_uses_ubuntu_runner(self) -> None:
        """All jobs must use ubuntu-latest (free-tier-safe)."""
        wf = _load_publish_workflow()
        jobs = wf.get("jobs", {})
        for job_name, job_config in jobs.items():
            runs_on = str(job_config.get("runs-on", ""))
            assert "ubuntu" in runs_on.lower(), (
                f"Job '{job_name}' should use ubuntu-latest for free-tier safety"
            )

    def test_uses_checkout(self) -> None:
        """Build job must checkout the repository."""
        content = _read(PUBLISH_WORKFLOW)
        assert "actions/checkout" in content, (
            "Publish workflow must use actions/checkout"
        )

    def test_uses_setup_python(self) -> None:
        """Build job must set up Python."""
        content = _read(PUBLISH_WORKFLOW)
        assert "actions/setup-python" in content, (
            "Publish workflow must use actions/setup-python"
        )

    def test_has_environment_configured(self) -> None:
        """Publish job should use a GitHub environment (recommended by PyPI docs)."""
        wf = _load_publish_workflow()
        jobs = wf.get("jobs", {})
        has_environment = False
        for job_config in jobs.values():
            if job_config.get("environment"):
                has_environment = True
                break
        assert has_environment, (
            "Publish workflow should configure a GitHub environment (recommended for Trusted Publishing)"
        )


# ===================================================================
# Free-tier safety
# ===================================================================


class TestPublishFreeTierSafety:
    """Publish workflow must be free for public repos."""

    def test_no_self_hosted_runners(self) -> None:
        content = _read(PUBLISH_WORKFLOW)
        assert "self-hosted" not in content.lower(), (
            "Publish workflow should not use self-hosted runners"
        )

    def test_no_paid_external_services(self) -> None:
        content = _read(PUBLISH_WORKFLOW).lower()
        for service in ["services:", "firestore", "redis", "postgres"]:
            assert service not in content, (
                f"Publish workflow should not reference external service: {service!r}"
            )
