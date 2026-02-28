"""Tests for memory benchmark matrix (Section 8.2).

Validates VAL-RUNBOOK-003:
- Memory section runs all required nvbandwidth tests
  (device_local_copy, device_local_copy_sm, H2D, D2H, D2D read, D2D write, D2D bidirectional)
- Plus PyTorch D2D cross-validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ornn_bench.models import BenchmarkStatus
from ornn_bench.runbook.memory import (
    REQUIRED_TESTS,
    MemoryMatrixRunner,
    collect_memory_matrix,
    run_nvbandwidth_test,
    run_pytorch_d2d_crossval,
)
from ornn_bench.runbook.parsers import parse_nvbandwidth_json

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(path: str) -> str:
    return (FIXTURES / path).read_text()


def _all_nvbw_fixtures() -> dict[str, str]:
    """Load fixture data for all required nvbandwidth tests."""
    return {
        "device_local_copy": _load_fixture("nvbandwidth/device_local_copy_ce.json"),
        "device_local_copy_sm": _load_fixture("nvbandwidth/device_local_copy_sm.json"),
        "h2d": _load_fixture("nvbandwidth/h2d.json"),
        "d2h": _load_fixture("nvbandwidth/d2h.json"),
        "d2d_read": _load_fixture("nvbandwidth/d2d_read.json"),
        "d2d_write": _load_fixture("nvbandwidth/d2d_write.json"),
        "d2d_bidir": _load_fixture("nvbandwidth/d2d_bidir.json"),
    }


def _mock_pytorch_result() -> dict[str, Any]:
    """Create mock PyTorch D2D cross-validation result."""
    return {
        "test_key": "pytorch_d2d_crossval",
        "bandwidth_gb_s": 1985.3,
        "size_bytes": 268435456,
        "iterations": 20,
    }


# ---------------------------------------------------------------------------
# nvbandwidth parser tests
# ---------------------------------------------------------------------------


class TestNvbandwidthParser:
    """Tests for the nvbandwidth JSON output parser."""

    def test_parses_testname(self) -> None:
        raw = _load_fixture("nvbandwidth/device_local_copy_ce.json")
        result = parse_nvbandwidth_json(raw)
        assert result["testname"] == "device_to_device_memcpy_read_ce"

    def test_parses_bandwidth_matrix(self) -> None:
        raw = _load_fixture("nvbandwidth/device_local_copy_ce.json")
        result = parse_nvbandwidth_json(raw)
        assert isinstance(result["bandwidth_matrix"], list)
        assert len(result["bandwidth_matrix"]) > 0

    def test_parses_sum_min_max(self) -> None:
        raw = _load_fixture("nvbandwidth/device_local_copy_ce.json")
        result = parse_nvbandwidth_json(raw)
        assert result["sum"] > 0
        assert result["min"] > 0
        assert result["max"] > 0

    def test_parses_h2d(self) -> None:
        raw = _load_fixture("nvbandwidth/h2d.json")
        result = parse_nvbandwidth_json(raw)
        assert result["testname"] == "host_to_device_memcpy_ce"
        assert result["max"] > 0

    def test_parses_d2h(self) -> None:
        raw = _load_fixture("nvbandwidth/d2h.json")
        result = parse_nvbandwidth_json(raw)
        assert result["testname"] == "device_to_host_memcpy_ce"

    def test_parses_d2d_write(self) -> None:
        raw = _load_fixture("nvbandwidth/d2d_write.json")
        result = parse_nvbandwidth_json(raw)
        assert "write" in result["testname"]

    def test_parses_d2d_bidir(self) -> None:
        raw = _load_fixture("nvbandwidth/d2d_bidir.json")
        result = parse_nvbandwidth_json(raw)
        assert "bidirectional" in result["testname"]


# ---------------------------------------------------------------------------
# Individual test runner tests
# ---------------------------------------------------------------------------


class TestRunNvbandwidthTest:
    """Tests for individual nvbandwidth test execution."""

    def test_h2d_with_fixture(self) -> None:
        raw = _load_fixture("nvbandwidth/h2d.json")
        result = run_nvbandwidth_test("h2d", raw_output=raw)
        assert result["test_key"] == "h2d"
        assert result["max"] > 0

    def test_d2d_read_with_fixture(self) -> None:
        raw = _load_fixture("nvbandwidth/d2d_read.json")
        result = run_nvbandwidth_test("d2d_read", raw_output=raw)
        assert result["test_key"] == "d2d_read"

    def test_d2d_bidir_with_fixture(self) -> None:
        raw = _load_fixture("nvbandwidth/d2d_bidir.json")
        result = run_nvbandwidth_test("d2d_bidir", raw_output=raw)
        assert result["test_key"] == "d2d_bidir"
        assert result["max"] == 4079.0

    def test_device_local_copy_sm_with_fixture(self) -> None:
        raw = _load_fixture("nvbandwidth/device_local_copy_sm.json")
        result = run_nvbandwidth_test("device_local_copy_sm", raw_output=raw)
        assert result["test_key"] == "device_local_copy_sm"
        assert "sm" in result["testname"]


# ---------------------------------------------------------------------------
# PyTorch cross-validation tests
# ---------------------------------------------------------------------------


class TestPytorchD2DCrossval:
    """Tests for PyTorch D2D cross-validation."""

    def test_mock_result_accepted(self) -> None:
        result = run_pytorch_d2d_crossval(raw_result=_mock_pytorch_result())
        assert result["test_key"] == "pytorch_d2d_crossval"
        assert result["bandwidth_gb_s"] > 0

    def test_mock_result_has_size_and_iters(self) -> None:
        result = run_pytorch_d2d_crossval(raw_result=_mock_pytorch_result())
        assert result["size_bytes"] > 0
        assert result["iterations"] > 0


# ---------------------------------------------------------------------------
# Full memory matrix tests
# ---------------------------------------------------------------------------


class TestCollectMemoryMatrix:
    """Tests for the full memory benchmark matrix collection."""

    def test_runs_all_required_tests(self) -> None:
        """VAL-RUNBOOK-003: All required nvbandwidth tests run."""
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert set(REQUIRED_TESTS) == set(result["tests_run"])

    def test_nvbandwidth_results_present_for_all_tests(self) -> None:
        """VAL-RUNBOOK-003: Results present for every required test."""
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        for test_name in REQUIRED_TESTS:
            assert test_name in result["nvbandwidth_results"]
            assert result["nvbandwidth_results"][test_name]["max"] > 0

    def test_includes_pytorch_crossval(self) -> None:
        """VAL-RUNBOOK-003: PyTorch D2D cross-validation included."""
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        crossval = result["pytorch_d2d_crossval"]
        assert crossval["test_key"] == "pytorch_d2d_crossval"
        assert crossval["bandwidth_gb_s"] > 0

    def test_device_local_copy_ce_present(self) -> None:
        """device_local_copy (CE) test present."""
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "device_local_copy" in result["nvbandwidth_results"]

    def test_device_local_copy_sm_present(self) -> None:
        """device_local_copy_sm test present."""
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "device_local_copy_sm" in result["nvbandwidth_results"]

    def test_h2d_present(self) -> None:
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "h2d" in result["nvbandwidth_results"]

    def test_d2h_present(self) -> None:
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "d2h" in result["nvbandwidth_results"]

    def test_d2d_read_present(self) -> None:
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "d2d_read" in result["nvbandwidth_results"]

    def test_d2d_write_present(self) -> None:
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "d2d_write" in result["nvbandwidth_results"]

    def test_d2d_bidir_present(self) -> None:
        result = collect_memory_matrix(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        assert "d2d_bidir" in result["nvbandwidth_results"]


# ---------------------------------------------------------------------------
# MemoryMatrixRunner section runner tests
# ---------------------------------------------------------------------------


class TestMemoryMatrixRunner:
    """Tests for the MemoryMatrixRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        runner = MemoryMatrixRunner(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        runner = MemoryMatrixRunner(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_name_is_memory(self) -> None:
        runner = MemoryMatrixRunner(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        result = runner.run()
        assert result.name == "memory"

    def test_runner_metrics_have_required_keys(self) -> None:
        runner = MemoryMatrixRunner(
            test_outputs=_all_nvbw_fixtures(),
            pytorch_d2d_result=_mock_pytorch_result(),
        )
        result = runner.run()
        assert "tests_run" in result.metrics
        assert "nvbandwidth_results" in result.metrics
        assert "pytorch_d2d_crossval" in result.metrics
