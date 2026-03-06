from __future__ import annotations

import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFRA_DIR = ROOT / "ornn-bench-infra"
TOOLCHAIN_ENV = INFRA_DIR / "toolchain.env"
PROVISION_SCRIPT = INFRA_DIR / "provision.sh"
BENCHMARK_NODE_SCRIPT = INFRA_DIR / "benchmark_node.sh"
PROVIDER_NOTES_DIR = INFRA_DIR / "provider-notes"

REQUIRED_TOOLCHAIN_VALUES = {
    "ORNN_BENCH_VERSION": "0.1.0",
    "MAMF_FINDER_REPO": "https://github.com/stas00/ml-engineering.git",
    "MAMF_FINDER_COMMIT": "7c660da71e533fdb5de141591379d3c8070ef272",
    "MAMF_FINDER_RELPATH": "compute/accelerator/benchmarks/mamf-finder.py",
    "NVBANDWIDTH_TAG": "v0.8",
    "NCCL_TESTS_COMMIT": "ae98985f5599617be94042f4aa3637d10014ce89",
    "NCCL_DEV_VERSION": "2.21.5-1+cuda12.4",
    "CUDA_VERSION": "12.4",
    "DRIVER_MINIMUM": "550",
    "PYTHON_VERSION": "3.10",
    "TORCH_VERSION": "2.5.1",
    "TORCH_CUDA_WHEEL": "cu124",
}
FULL_SHA_KEYS = {"MAMF_FINDER_COMMIT", "NCCL_TESTS_COMMIT"}
PROVIDER_FILES = {
    "aws.md": "AWS",
    "gcp.md": "GCP",
    "azure.md": "Azure",
    "oracle.md": "Oracle",
}
SCRIPT_PATHS = (PROVISION_SCRIPT, BENCHMARK_NODE_SCRIPT)


def _read(path: Path) -> str:
    assert path.exists(), f"Expected file not found: {path}"
    return path.read_text(encoding="utf-8")


def _parse_toolchain_env() -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in _read(TOOLCHAIN_ENV).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r'([A-Z0-9_]+)="([^"]*)"', line)
        assert match is not None, f"Expected KEY=\"VALUE\" entry, got: {raw_line!r}"
        key, value = match.groups()
        parsed[key] = value
    return parsed


class TestBenchInfraLayout:
    def test_required_top_level_files_exist(self) -> None:
        for path in (TOOLCHAIN_ENV, PROVISION_SCRIPT, BENCHMARK_NODE_SCRIPT):
            assert path.exists(), f"Missing required bench infra file: {path}"

    def test_provider_notes_exist_for_primary_clouds(self) -> None:
        assert PROVIDER_NOTES_DIR.is_dir(), "provider-notes directory is missing"
        for filename, provider_name in PROVIDER_FILES.items():
            note_path = PROVIDER_NOTES_DIR / filename
            content = _read(note_path)
            assert provider_name.lower() in content.lower()
            assert "placeholder" in content.lower()


class TestToolchainEnv:
    def test_toolchain_env_declares_immutability(self) -> None:
        content = _read(TOOLCHAIN_ENV).lower()
        assert "immutable at runtime" in content
        assert "must not modify or append" in content

    def test_toolchain_env_has_required_pin_values(self) -> None:
        parsed = _parse_toolchain_env()
        for key, expected_value in REQUIRED_TOOLCHAIN_VALUES.items():
            assert parsed.get(key) == expected_value, f"Unexpected value for {key}"

    def test_toolchain_env_uses_full_shas_for_repo_pins(self) -> None:
        parsed = _parse_toolchain_env()
        for key in FULL_SHA_KEYS:
            assert re.fullmatch(r"[0-9a-f]{40}", parsed[key])


class TestShellScripts:
    def test_scripts_use_strict_bash_headers(self) -> None:
        for path in SCRIPT_PATHS:
            lines = _read(path).splitlines()
            assert lines[0] in {"#!/usr/bin/env bash", "#!/bin/bash"}
            assert "set -euo pipefail" in lines[:5]

    def test_scripts_are_executable(self) -> None:
        for path in SCRIPT_PATHS:
            mode = path.stat().st_mode
            assert mode & stat.S_IXUSR, f"Expected executable bit on {path}"

    def test_scripts_do_not_mutate_toolchain_lock(self) -> None:
        forbidden_patterns = (
            r">>?.*toolchain\.env",
            r"tee\s+.*toolchain\.env",
            r"sed\s+-i.*toolchain\.env",
        )
        for path in SCRIPT_PATHS:
            content = _read(path)
            for pattern in forbidden_patterns:
                assert re.search(pattern, content) is None, (
                    f"Script should not mutate toolchain.env: {path} matched {pattern!r}"
                )


class TestProvisionScript:
    def test_sources_toolchain_lock_and_writes_runtime_env(self) -> None:
        content = _read(PROVISION_SCRIPT)

        assert 'source "${TOOLCHAIN_ENV}"' in content
        assert 'readonly RUNTIME_ENV="${TOOLS_ROOT}/runtime.env"' in content
        assert 'write_runtime_env() {' in content
        assert 'mv "${temp_env}" "${RUNTIME_ENV}"' in content

    def test_installs_and_verifies_pinned_toolchain_components(self) -> None:
        content = _read(PROVISION_SCRIPT)

        required_snippets = (
            '"ornn-bench==${ORNN_BENCH_VERSION}"',
            '"torch==${TORCH_VERSION}"',
            'https://download.pytorch.org/whl/${TORCH_CUDA_WHEEL}',
            'sync_git_checkout "${MAMF_FINDER_REPO}" "${MAMF_REPO_DIR}" "${MAMF_FINDER_COMMIT}"',
            'verify_git_ref "${MAMF_REPO_DIR}" "${MAMF_FINDER_COMMIT}" "mamf-finder repo"',
            'sync_git_checkout "${NVBANDWIDTH_REPO}" "${NVBANDWIDTH_REPO_DIR}" "${NVBANDWIDTH_TAG}"',
            'grep -Fxq "${NVBANDWIDTH_TAG}"',
            'sync_git_checkout "${NCCL_TESTS_REPO}" "${NCCL_TESTS_REPO_DIR}" "${NCCL_TESTS_COMMIT}"',
            'verify_git_ref "${NCCL_TESTS_REPO_DIR}" "${NCCL_TESTS_COMMIT}" "nccl-tests repo"',
            '"libnccl2=${NCCL_DEV_VERSION}"',
            '"libnccl-dev=${NCCL_DEV_VERSION}"',
        )

        for snippet in required_snippets:
            assert snippet in content, f"Expected pinned install snippet: {snippet}"

    def test_hard_fails_when_pinned_nccl_packages_are_unavailable(self) -> None:
        content = _read(PROVISION_SCRIPT)

        assert 'ensure_apt_package_version_available() {' in content
        assert 'apt-cache madison "${package_name}"' in content
        assert 'Pinned ${package_label} package ${package_name}=${version} is unavailable.' in content
        assert 'Ensure the NVIDIA CUDA apt repository for CUDA ${CUDA_VERSION} is configured' in content

        assert (
            'ensure_apt_package_version_available "libnccl2" '
            '"${NCCL_DEV_VERSION}" "NCCL runtime"' in content
        )
        assert (
            'ensure_apt_package_version_available "libnccl-dev" '
            '"${NCCL_DEV_VERSION}" "NCCL development"' in content
        )

    def test_runtime_env_exports_discovered_tool_paths(self) -> None:
        content = _read(PROVISION_SCRIPT)

        exported_keys = (
            'ORNN_BENCH_BIN',
            'MAMF_FINDER_PATH',
            'NVBANDWIDTH_PATH',
            'ALL_REDUCE_PERF_PATH',
            'REDUCE_SCATTER_PERF_PATH',
            'ALL_GATHER_PERF_PATH',
            'BROADCAST_PERF_PATH',
            'SENDRECV_PERF_PATH',
            'CUDA_HOME',
            'NCCL_LIB_DIR',
        )

        for key in exported_keys:
            assert f"printf '{key}=\"%s\"\\n'" in content
