"""Tests for interconnect benchmark matrix (Section 8.3).

Validates VAL-RUNBOOK-004:
- Interconnect section runs all required NCCL tests
  (all-reduce sweep, 1GB all-reduce variance, reduce-scatter,
   all-gather, broadcast, send-receive)
- Captures bus bandwidth outputs
"""

from __future__ import annotations

from pathlib import Path

from ornn_bench.models import BenchmarkStatus
from ornn_bench.runbook.interconnect import (
    REQUIRED_NCCL_TESTS,
    InterconnectMatrixRunner,
    collect_interconnect_matrix,
    run_nccl_test,
)
from ornn_bench.runbook.parsers import parse_nccl_output

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(path: str) -> str:
    return (FIXTURES / path).read_text()


def _all_nccl_fixtures() -> dict[str, str]:
    """Load fixture data for all required NCCL tests."""
    return {
        "all_reduce_sweep": _load_fixture("nccl_tests/all_reduce_sweep.txt"),
        "all_reduce_1gb": _load_fixture("nccl_tests/all_reduce_1gb.txt"),
        "reduce_scatter": _load_fixture("nccl_tests/reduce_scatter.txt"),
        "all_gather": _load_fixture("nccl_tests/all_gather.txt"),
        "broadcast": _load_fixture("nccl_tests/broadcast.txt"),
        "sendrecv": _load_fixture("nccl_tests/sendrecv.txt"),
    }


# ---------------------------------------------------------------------------
# nccl-tests parser tests
# ---------------------------------------------------------------------------


class TestNcclParser:
    """Tests for the nccl-tests output parser."""

    def test_parses_devices(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = parse_nccl_output(raw)
        assert len(result["devices"]) == 2
        for dev in result["devices"]:
            assert "rank" in dev
            assert "gpu_name" in dev
            assert "H100" in dev["gpu_name"]

    def test_parses_data_rows(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) > 0

    def test_data_rows_have_bandwidth(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = parse_nccl_output(raw)
        for row in result["results"]:
            assert "out_of_place" in row
            assert "in_place" in row
            assert "busbw" in row["out_of_place"]
            assert "busbw" in row["in_place"]

    def test_parses_avg_bus_bandwidth(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = parse_nccl_output(raw)
        assert result["avg_bus_bandwidth"] == 8.39

    def test_computes_max_busbw(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = parse_nccl_output(raw)
        assert result["max_busbw"] > 0

    def test_parses_1gb_fixed_run(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_1gb.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) == 1
        assert result["avg_bus_bandwidth"] == 18.54

    def test_parses_reduce_scatter(self) -> None:
        raw = _load_fixture("nccl_tests/reduce_scatter.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) >= 1
        assert result["avg_bus_bandwidth"] > 0

    def test_parses_all_gather(self) -> None:
        raw = _load_fixture("nccl_tests/all_gather.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) >= 1

    def test_parses_broadcast(self) -> None:
        raw = _load_fixture("nccl_tests/broadcast.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) >= 1

    def test_parses_sendrecv(self) -> None:
        raw = _load_fixture("nccl_tests/sendrecv.txt")
        result = parse_nccl_output(raw)
        assert len(result["results"]) >= 1

    def test_empty_input_returns_empty(self) -> None:
        result = parse_nccl_output("")
        assert result["devices"] == []
        assert result["results"] == []
        assert result["avg_bus_bandwidth"] == 0.0


# ---------------------------------------------------------------------------
# Individual test runner tests
# ---------------------------------------------------------------------------


class TestRunNcclTest:
    """Tests for individual NCCL test execution."""

    def test_all_reduce_sweep_with_fixture(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = run_nccl_test("all_reduce_sweep", raw_output=raw)
        assert result["test_key"] == "all_reduce_sweep"
        assert result["avg_bus_bandwidth"] > 0

    def test_all_reduce_1gb_with_fixture(self) -> None:
        raw = _load_fixture("nccl_tests/all_reduce_1gb.txt")
        result = run_nccl_test("all_reduce_1gb", raw_output=raw)
        assert result["test_key"] == "all_reduce_1gb"
        assert result["avg_bus_bandwidth"] == 18.54

    def test_reduce_scatter_with_fixture(self) -> None:
        raw = _load_fixture("nccl_tests/reduce_scatter.txt")
        result = run_nccl_test("reduce_scatter", raw_output=raw)
        assert result["test_key"] == "reduce_scatter"

    def test_unknown_test_returns_error(self) -> None:
        result = run_nccl_test("nonexistent_test")
        assert "error" in result
        assert result["avg_bus_bandwidth"] == 0.0

    def test_bus_bandwidth_captured(self) -> None:
        """VAL-RUNBOOK-004: Bus bandwidth captured."""
        raw = _load_fixture("nccl_tests/all_reduce_sweep.txt")
        result = run_nccl_test("all_reduce_sweep", raw_output=raw)
        assert result["max_busbw"] > 0


# ---------------------------------------------------------------------------
# Full interconnect matrix tests
# ---------------------------------------------------------------------------


class TestCollectInterconnectMatrix:
    """Tests for the full interconnect benchmark matrix collection."""

    def test_runs_all_required_tests(self) -> None:
        """VAL-RUNBOOK-004: All required NCCL tests run."""
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert set(REQUIRED_NCCL_TESTS) == set(result["tests_run"])

    def test_all_reduce_sweep_present(self) -> None:
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "all_reduce_sweep" in result["nccl_results"]

    def test_all_reduce_1gb_present(self) -> None:
        """VAL-RUNBOOK-004: 1GB all-reduce variance test present."""
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "all_reduce_1gb" in result["nccl_results"]

    def test_reduce_scatter_present(self) -> None:
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "reduce_scatter" in result["nccl_results"]

    def test_all_gather_present(self) -> None:
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "all_gather" in result["nccl_results"]

    def test_broadcast_present(self) -> None:
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "broadcast" in result["nccl_results"]

    def test_sendrecv_present(self) -> None:
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "sendrecv" in result["nccl_results"]

    def test_bus_bandwidth_summary_present(self) -> None:
        """VAL-RUNBOOK-004: Bus bandwidth summary captured."""
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        assert "bus_bandwidth_summary" in result
        for test_name in REQUIRED_NCCL_TESTS:
            assert test_name in result["bus_bandwidth_summary"]
            bw_entry = result["bus_bandwidth_summary"][test_name]
            assert "avg_busbw" in bw_entry
            assert "max_busbw" in bw_entry

    def test_nccl_results_have_device_info(self) -> None:
        """NCCL results include device information."""
        result = collect_interconnect_matrix(test_outputs=_all_nccl_fixtures())
        for test_name in REQUIRED_NCCL_TESTS:
            test_result = result["nccl_results"][test_name]
            assert len(test_result["devices"]) > 0


# ---------------------------------------------------------------------------
# InterconnectMatrixRunner section runner tests
# ---------------------------------------------------------------------------


class TestInterconnectMatrixRunner:
    """Tests for the InterconnectMatrixRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        runner = InterconnectMatrixRunner(test_outputs=_all_nccl_fixtures())
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        runner = InterconnectMatrixRunner(test_outputs=_all_nccl_fixtures())
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_name_is_interconnect(self) -> None:
        runner = InterconnectMatrixRunner(test_outputs=_all_nccl_fixtures())
        result = runner.run()
        assert result.name == "interconnect"

    def test_runner_metrics_have_required_keys(self) -> None:
        runner = InterconnectMatrixRunner(test_outputs=_all_nccl_fixtures())
        result = runner.run()
        assert "tests_run" in result.metrics
        assert "nccl_results" in result.metrics
        assert "bus_bandwidth_summary" in result.metrics
