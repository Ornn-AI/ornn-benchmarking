"""Tests for deployment configuration constraints.

Validates that deployment scripts/configs reference only approved services
(Cloud Run + Firestore), enforce scale-to-zero defaults, and document
local testing guidance including the Firestore emulator Java prerequisite
and mock fallback path.

Fulfills:
    VAL-CROSS-004 — Default deployment path uses Cloud Run + Firestore only
    VAL-DEPLOY-006 — Deployment uses free-tier-safe configuration
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = ROOT / "deploy.sh"
DOCKERFILE = ROOT / "Dockerfile"
DEPLOY_DOCS = ROOT / "docs" / "deployment.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    """Read file content, raising a clear error if missing."""
    assert path.exists(), f"Expected file not found: {path}"
    return path.read_text(encoding="utf-8")


# ===================================================================
# Deploy script: Cloud Run + Firestore only
# ===================================================================


class TestDeployScriptExists:
    """Deployment script must exist and be executable."""

    def test_deploy_script_exists(self) -> None:
        assert DEPLOY_SCRIPT.exists(), "deploy.sh not found at project root"

    def test_deploy_script_is_shell(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        assert content.startswith("#!/"), "deploy.sh should have a shebang line"


class TestDeployScriptApprovedServicesOnly:
    """Deploy script must reference only Cloud Run + Firestore — no paid infra."""

    # Patterns for forbidden GCP services that violate free-tier constraint
    FORBIDDEN_SERVICES: ClassVar[list[tuple[str, str]]] = [
        (r"\bgke\b", "GKE (Kubernetes Engine)"),
        (r"\bcloud[\s_-]*sql\b", "Cloud SQL"),
        (r"\bcompute[\s_-]*instances?\b", "Compute Engine VMs"),
        (r"\bapp[\s_-]*engine\b", "App Engine"),
        (r"\bcloud[\s_-]*functions\b", "Cloud Functions"),
        (r"\bkubectl\b", "kubectl (implies GKE)"),
        (r"\bcloud[\s_-]*memorystore\b", "Memorystore (Redis)"),
    ]

    def test_no_forbidden_services_in_deploy_script(self) -> None:
        content = _read(DEPLOY_SCRIPT).lower()
        for pattern, service_name in self.FORBIDDEN_SERVICES:
            matches = re.findall(pattern, content)
            assert not matches, (
                f"deploy.sh references forbidden service: {service_name} "
                f"(pattern: {pattern!r}, matches: {matches})"
            )

    def test_references_cloud_run(self) -> None:
        content = _read(DEPLOY_SCRIPT).lower()
        assert "cloud run" in content or "gcloud run" in content, (
            "deploy.sh should reference Cloud Run"
        )

    def test_references_firestore(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        assert "FIRESTORE" in content or "firestore" in content, (
            "deploy.sh should reference Firestore configuration"
        )


class TestDeployScriptScaleToZero:
    """Cloud Run must be configured with min-instances=0 (scale-to-zero)."""

    def test_min_instances_is_zero(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        # Look for --min-instances=0 or MIN_INSTANCES=0
        assert re.search(r"min.instances[\"']?[=\s]+[\"']?0", content, re.IGNORECASE), (
            "deploy.sh must set min-instances to 0 for scale-to-zero"
        )

    def test_uses_managed_platform(self) -> None:
        content = _read(DEPLOY_SCRIPT).lower()
        assert "managed" in content, (
            "deploy.sh should use --platform=managed (not GKE)"
        )

    def test_sets_firestore_project_env_var(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        assert "FIRESTORE_PROJECT_ID" in content, (
            "deploy.sh should pass FIRESTORE_PROJECT_ID to the Cloud Run service"
        )


# ===================================================================
# Dockerfile
# ===================================================================


class TestDockerfileConstraints:
    """Dockerfile should be minimal and appropriate for Cloud Run."""

    def test_dockerfile_exists(self) -> None:
        assert DOCKERFILE.exists(), "Dockerfile not found at project root"

    def test_uses_python_base_image(self) -> None:
        content = _read(DOCKERFILE)
        assert re.search(r"FROM\s+python:", content), (
            "Dockerfile should use a Python base image"
        )

    def test_copies_api_source(self) -> None:
        content = _read(DOCKERFILE).lower()
        assert "copy" in content and "api" in content, (
            "Dockerfile should copy the API source code"
        )

    def test_exposes_port(self) -> None:
        content = _read(DOCKERFILE)
        assert "EXPOSE" in content or "PORT" in content, (
            "Dockerfile should expose or reference a port"
        )

    def test_runs_uvicorn(self) -> None:
        content = _read(DOCKERFILE).lower()
        assert "uvicorn" in content, (
            "Dockerfile should run uvicorn to serve the FastAPI app"
        )

    def test_no_gpu_dependencies(self) -> None:
        """Dockerfile is for the API, not the benchmark runner — no CUDA/GPU."""
        content = _read(DOCKERFILE).lower()
        for forbidden in ["nvidia", "cuda", "torch", "gpu"]:
            assert forbidden not in content, (
                f"Dockerfile should not include GPU dependency: {forbidden}"
            )


# ===================================================================
# Deployment documentation
# ===================================================================


class TestDeploymentDocsExist:
    """Deployment documentation must exist."""

    def test_deployment_docs_exist(self) -> None:
        assert DEPLOY_DOCS.exists(), "docs/deployment.md not found"

    def test_deployment_docs_not_empty(self) -> None:
        content = _read(DEPLOY_DOCS)
        assert len(content) > 500, (
            "docs/deployment.md seems too short to be comprehensive"
        )


class TestDeploymentDocsContent:
    """Deployment docs must cover key topics."""

    def test_documents_cloud_run(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "cloud run" in content, (
            "Deployment docs should mention Cloud Run"
        )

    def test_documents_firestore(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "firestore" in content, (
            "Deployment docs should mention Firestore"
        )

    def test_documents_scale_to_zero(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "scale" in content and "zero" in content, (
            "Deployment docs should explain scale-to-zero behavior"
        )

    def test_documents_no_paid_infra(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        # Docs should explicitly call out that paid infra is not used
        assert "free" in content or "no always-on" in content or "zero cost" in content, (
            "Deployment docs should mention free-tier-safe or no always-on infra"
        )


class TestFirestoreEmulatorJavaPrerequisite:
    """Docs must document Firestore emulator Java prerequisite."""

    def test_mentions_java_prerequisite(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "java" in content, (
            "Deployment docs must mention Java as a prerequisite for Firestore emulator"
        )

    def test_mentions_jdk_version(self) -> None:
        content = _read(DEPLOY_DOCS)
        # Should mention JDK 11+ or similar version requirement
        assert re.search(r"(?:jdk|java)\s*\d+", content, re.IGNORECASE), (
            "Deployment docs should specify a Java/JDK version requirement"
        )

    def test_mentions_emulator_command(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "gcloud emulators firestore" in content or "firestore emulator" in content, (
            "Deployment docs should mention how to start the Firestore emulator"
        )


class TestMockFallbackPath:
    """Docs must document the mock/fallback test server path."""

    def test_mentions_mock_fallback(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "mock" in content, (
            "Deployment docs should mention mock fallback for testing without Firestore"
        )

    def test_mentions_testing_without_java(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        # Should explain how to test when Java/emulator is unavailable
        assert ("without java" in content
                or "without the emulator" in content
                or "no java" in content
                or "without java" in content
                or "machines without java" in content
                or ("mock" in content and "fallback" in content)), (
            "Deployment docs should explain testing path when Java is unavailable"
        )

    def test_mentions_in_memory_or_mock_client(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "in-memory" in content or "mock" in content, (
            "Deployment docs should mention in-memory mock or test client"
        )

    def test_mentions_ci_testing_path(self) -> None:
        content = _read(DEPLOY_DOCS).lower()
        assert "ci" in content, (
            "Deployment docs should mention CI/CD testing path"
        )


# ===================================================================
# Cross-validation: services manifest consistency
# ===================================================================


class TestServicesManifestConsistency:
    """Deployment config should be consistent with .factory/services.yaml."""

    def test_api_port_matches_manifest(self) -> None:
        """deploy.sh port should match the manifest port for API service."""
        deploy_content = _read(DEPLOY_SCRIPT)
        # The deploy script should reference port 8080 (matching services.yaml)
        assert "8080" in deploy_content, (
            "deploy.sh should reference port 8080 matching services.yaml"
        )

    def test_firestore_emulator_port_in_docs(self) -> None:
        """Deployment docs should reference port 8085 for Firestore emulator."""
        docs_content = _read(DEPLOY_DOCS)
        assert "8085" in docs_content, (
            "Deployment docs should reference Firestore emulator port 8085"
        )


# ===================================================================
# Free-tier safety: no always-on services
# ===================================================================


class TestFreeTierSafety:
    """End-to-end free-tier safety validation."""

    ALWAYS_ON_INDICATORS: ClassVar[list[str]] = [
        "always-on",
        "reserved instance",
        "committed use",
        "min-instances=1",
        "min-instances: 1",
        "MIN_INSTANCES=1",
    ]

    def test_deploy_script_no_always_on_config(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        for indicator in self.ALWAYS_ON_INDICATORS:
            assert indicator not in content, (
                f"deploy.sh contains always-on indicator: {indicator!r}"
            )

    def test_dockerfile_no_always_on_config(self) -> None:
        content = _read(DOCKERFILE)
        for indicator in self.ALWAYS_ON_INDICATORS:
            assert indicator not in content, (
                f"Dockerfile contains always-on indicator: {indicator!r}"
            )

    def test_default_project_is_ornn_benchmarking(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        assert "ornn-benchmarking" in content, (
            "deploy.sh default project should be ornn-benchmarking"
        )

    def test_default_region_is_us_east1(self) -> None:
        content = _read(DEPLOY_SCRIPT)
        assert "us-east1" in content, (
            "deploy.sh default region should be us-east1 (matching Firestore)"
        )
