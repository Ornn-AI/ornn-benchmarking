"""Tests for pre-flight inventory capture (Section 8.0).

Validates VAL-RUNBOOK-001:
- Pre-flight captures GPU UUIDs, NVLink topology/status
- Driver/CUDA/software versions captured
- OS/kernel, CPU/NUMA topology captured
- Baseline NVLink/ECC/XID-related diagnostics captured
"""

from __future__ import annotations

from pathlib import Path

from ornn_bench.models import BenchmarkStatus
from ornn_bench.runbook.parsers import parse_nvidia_smi_q
from ornn_bench.runbook.preflight import (
    PreflightRunner,
    collect_preflight_inventory,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(path: str) -> str:
    return (FIXTURES / path).read_text()


# ---------------------------------------------------------------------------
# nvidia-smi -q parser tests
# ---------------------------------------------------------------------------


class TestNvidiaSmiQParser:
    """Tests for the nvidia-smi -q output parser."""

    def test_parses_driver_version(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        assert result["driver_version"] == "535.129.03"

    def test_parses_cuda_version(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        assert result["cuda_version"] == "12.2"

    def test_parses_attached_gpus(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        assert result["attached_gpus"] == 2

    def test_parses_gpu_uuids(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        uuids = [g["uuid"] for g in result["gpus"]]
        assert "GPU-12345678-abcd-1234-abcd-123456789abc" in uuids
        assert "GPU-87654321-dcba-4321-dcba-cba987654321" in uuids

    def test_parses_gpu_product_name(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        names = [g["product_name"] for g in result["gpus"]]
        assert all("H100" in name for name in names)

    def test_parses_ecc_mode(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            assert gpu["ecc_mode"] == "Enabled"

    def test_parses_ecc_errors(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            errors = gpu["ecc_errors"]
            assert isinstance(errors, dict)
            # All errors should be 0 in our fixture
            for val in errors.values():
                assert val == 0

    def test_parses_nvlink_entries(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            assert len(gpu["nvlink"]) > 0
            for link in gpu["nvlink"]:
                assert "link_id" in link
                assert link["state"] == "Active"
                assert "remote_gpu_uuid" in link

    def test_parses_memory_total(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            assert gpu["memory_total_mib"] == 81920

    def test_parses_temperature(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            assert gpu["temperature_gpu_c"] > 0

    def test_parses_power(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        result = parse_nvidia_smi_q(raw)
        for gpu in result["gpus"]:
            assert gpu["power_draw_w"] > 0
            assert gpu["power_limit_w"] > 0


# ---------------------------------------------------------------------------
# Pre-flight inventory collection tests
# ---------------------------------------------------------------------------


class TestPreflightInventory:
    """Tests for the full pre-flight inventory collection."""

    def test_captures_gpu_uuids(self) -> None:
        """VAL-RUNBOOK-001: GPU UUIDs captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        uuids = inv["gpu_inventory"]["gpu_uuids"]
        assert len(uuids) == 2
        assert "GPU-12345678-abcd-1234-abcd-123456789abc" in uuids

    def test_captures_nvlink_topology(self) -> None:
        """VAL-RUNBOOK-001: NVLink topology/status captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert len(inv["nvlink_topology"]) > 0
        for link in inv["nvlink_topology"]:
            assert "state" in link

    def test_captures_driver_version(self) -> None:
        """VAL-RUNBOOK-001: Driver version captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert inv["driver_version"] == "535.129.03"

    def test_captures_cuda_version(self) -> None:
        """VAL-RUNBOOK-001: CUDA version captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert inv["cuda_version"] == "12.2"

    def test_captures_software_versions(self) -> None:
        """VAL-RUNBOOK-001: Software versions captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert "python" in inv["software_versions"]
        assert "driver" in inv["software_versions"]
        assert "cuda" in inv["software_versions"]

    def test_captures_os_kernel(self) -> None:
        """VAL-RUNBOOK-001: OS/kernel captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert inv["os"] != ""
        assert inv["kernel"] != ""

    def test_captures_cpu_model(self) -> None:
        """VAL-RUNBOOK-001: CPU model captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert inv["cpu_model"] != ""

    def test_captures_ecc_baseline(self) -> None:
        """VAL-RUNBOOK-001: Baseline ECC diagnostics captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        assert isinstance(inv["ecc_baseline"], dict)
        # Should have entries for each GPU UUID
        assert len(inv["ecc_baseline"]) > 0

    def test_captures_gpu_details(self) -> None:
        """VAL-RUNBOOK-001: Per-GPU identity details captured."""
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        inv = collect_preflight_inventory(nvidia_smi_q_output=raw)
        gpus = inv["gpu_inventory"]["gpus"]
        assert len(gpus) == 2
        for gpu in gpus:
            assert gpu["uuid"] != ""
            assert gpu["product_name"] != ""
            assert gpu["serial_number"] != ""


# ---------------------------------------------------------------------------
# PreflightRunner tests
# ---------------------------------------------------------------------------


class TestPreflightRunner:
    """Tests for the PreflightRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        runner = PreflightRunner(nvidia_smi_q_output=raw)
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        runner = PreflightRunner(nvidia_smi_q_output=raw)
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_has_metrics(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        runner = PreflightRunner(nvidia_smi_q_output=raw)
        result = runner.run()
        assert "gpu_inventory" in result.metrics
        assert "driver_version" in result.metrics

    def test_runner_name_is_preflight(self) -> None:
        raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        runner = PreflightRunner(nvidia_smi_q_output=raw)
        result = runner.run()
        assert result.name == "pre-flight"
