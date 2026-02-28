"""Tests for monitoring capture and post-flight consistency checks.

Validates:
- VAL-RUNBOOK-005: Monitoring during benchmark execution
  Thermal/power monitoring runs continuously during benchmark phases,
  captures pre/post snapshots plus XID detection output.

- VAL-RUNBOOK-006: Post-flight consistency checks
  Post-flight checks validate seal/close UUID consistency and collect
  post-run NVLink/ECC error status with explicit pass/fail reporting.
"""

from __future__ import annotations

from pathlib import Path

from ornn_bench.models import BenchmarkStatus
from ornn_bench.runbook.monitoring import (
    MonitoringRunner,
    collect_monitoring_data,
    parse_dmon_output,
    parse_xid_errors,
)
from ornn_bench.runbook.postflight import (
    PostflightRunner,
    check_ecc_errors,
    check_nvlink_status,
    check_uuid_consistency,
    collect_postflight_checks,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(path: str) -> str:
    return (FIXTURES / path).read_text()


# ---------------------------------------------------------------------------
# nvidia-smi dmon parser tests
# ---------------------------------------------------------------------------


class TestDmonParser:
    """Tests for the nvidia-smi dmon output parser."""

    def test_parses_gpu_entries(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        assert len(result) > 0

    def test_entry_has_gpu_index(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "gpu_index" in entry

    def test_entry_has_power_watts(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "power_w" in entry
            assert entry["power_w"] > 0

    def test_entry_has_temperature(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "gpu_temp_c" in entry
            assert entry["gpu_temp_c"] > 0

    def test_entry_has_sm_utilization(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "sm_util_pct" in entry

    def test_entry_has_memory_utilization(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "mem_util_pct" in entry

    def test_entry_has_clocks(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        for entry in result:
            assert "mem_clock_mhz" in entry
            assert "gpu_clock_mhz" in entry

    def test_filters_by_gpu_index(self) -> None:
        raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = parse_dmon_output(raw)
        gpu0_entries = [e for e in result if e["gpu_index"] == 0]
        gpu1_entries = [e for e in result if e["gpu_index"] == 1]
        assert len(gpu0_entries) > 0
        assert len(gpu1_entries) > 0

    def test_empty_input(self) -> None:
        result = parse_dmon_output("")
        assert result == []

    def test_comments_only_input(self) -> None:
        result = parse_dmon_output("# header\n# another header\n")
        assert result == []


# ---------------------------------------------------------------------------
# XID error parser tests
# ---------------------------------------------------------------------------


class TestXidErrorParser:
    """Tests for XID error detection from dmesg/log output."""

    def test_detects_xid_errors(self) -> None:
        raw = _load_fixture("nvidia_smi/xid_errors.txt")
        result = parse_xid_errors(raw)
        assert len(result) > 0

    def test_xid_entry_has_error_code(self) -> None:
        raw = _load_fixture("nvidia_smi/xid_errors.txt")
        result = parse_xid_errors(raw)
        for entry in result:
            assert "xid_code" in entry
            assert isinstance(entry["xid_code"], int)

    def test_xid_entry_has_pci_id(self) -> None:
        raw = _load_fixture("nvidia_smi/xid_errors.txt")
        result = parse_xid_errors(raw)
        for entry in result:
            assert "pci_id" in entry

    def test_no_xid_errors(self) -> None:
        raw = _load_fixture("nvidia_smi/xid_errors_none.txt")
        result = parse_xid_errors(raw)
        assert result == []

    def test_parses_specific_codes(self) -> None:
        raw = _load_fixture("nvidia_smi/xid_errors.txt")
        result = parse_xid_errors(raw)
        codes = [e["xid_code"] for e in result]
        assert 48 in codes
        assert 63 in codes


# ---------------------------------------------------------------------------
# Monitoring data collection tests
# ---------------------------------------------------------------------------


class TestCollectMonitoringData:
    """Tests for full monitoring data collection."""

    def test_contains_pre_snapshot(self) -> None:
        """VAL-RUNBOOK-005: Pre snapshot captured."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = collect_monitoring_data(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        assert "pre_snapshot" in result
        assert result["pre_snapshot"]["driver_version"] == "535.129.03"

    def test_contains_post_snapshot(self) -> None:
        """VAL-RUNBOOK-005: Post snapshot captured."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = collect_monitoring_data(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        assert "post_snapshot" in result
        assert result["post_snapshot"]["driver_version"] == "535.129.03"

    def test_contains_time_series(self) -> None:
        """VAL-RUNBOOK-005: Continuous monitoring time-series captured."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = collect_monitoring_data(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        assert "time_series" in result
        assert len(result["time_series"]) > 0

    def test_contains_xid_detection(self) -> None:
        """VAL-RUNBOOK-005: XID detection output captured."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        xid_raw = _load_fixture("nvidia_smi/xid_errors.txt")
        result = collect_monitoring_data(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw=xid_raw,
        )
        assert "xid_errors" in result
        assert len(result["xid_errors"]) > 0

    def test_xid_clear_when_no_errors(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        result = collect_monitoring_data(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        assert result["xid_errors"] == []


# ---------------------------------------------------------------------------
# MonitoringRunner tests
# ---------------------------------------------------------------------------


class TestMonitoringRunner:
    """Tests for the MonitoringRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        runner = MonitoringRunner(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        runner = MonitoringRunner(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_has_monitoring_metrics(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        dmon_raw = _load_fixture("nvidia_smi/dmon_output.txt")
        runner = MonitoringRunner(
            pre_snapshot_raw=pre_raw,
            post_snapshot_raw=post_raw,
            dmon_raw=dmon_raw,
            xid_raw="",
        )
        result = runner.run()
        assert "pre_snapshot" in result.metrics
        assert "post_snapshot" in result.metrics
        assert "time_series" in result.metrics
        assert "xid_errors" in result.metrics

    def test_runner_name_is_monitoring(self) -> None:
        runner = MonitoringRunner(
            pre_snapshot_raw="",
            post_snapshot_raw="",
            dmon_raw="",
            xid_raw="",
        )
        result = runner.run()
        assert result.name == "monitoring"


# ---------------------------------------------------------------------------
# Post-flight UUID consistency check tests
# ---------------------------------------------------------------------------


class TestUUIDConsistency:
    """Tests for UUID consistency verification."""

    def test_consistent_uuids_pass(self) -> None:
        """VAL-RUNBOOK-006: UUID consistency with matching sets."""
        pre_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
            "GPU-87654321-dcba-4321-dcba-cba987654321",
        ]
        post_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
            "GPU-87654321-dcba-4321-dcba-cba987654321",
        ]
        result = check_uuid_consistency(pre_uuids, post_uuids)
        assert result["passed"] is True
        assert result["missing_uuids"] == []
        assert result["new_uuids"] == []

    def test_missing_uuid_fails(self) -> None:
        """VAL-RUNBOOK-006: Missing UUID causes failure."""
        pre_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
            "GPU-87654321-dcba-4321-dcba-cba987654321",
        ]
        post_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
        ]
        result = check_uuid_consistency(pre_uuids, post_uuids)
        assert result["passed"] is False
        assert "GPU-87654321-dcba-4321-dcba-cba987654321" in result["missing_uuids"]

    def test_new_uuid_fails(self) -> None:
        """VAL-RUNBOOK-006: New UUID appearing post-run causes failure."""
        pre_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
        ]
        post_uuids = [
            "GPU-12345678-abcd-1234-abcd-123456789abc",
            "GPU-new-uuid-000-000-000000000000",
        ]
        result = check_uuid_consistency(pre_uuids, post_uuids)
        assert result["passed"] is False
        assert "GPU-new-uuid-000-000-000000000000" in result["new_uuids"]

    def test_empty_pre_and_post_passes(self) -> None:
        result = check_uuid_consistency([], [])
        assert result["passed"] is True


# ---------------------------------------------------------------------------
# Post-flight NVLink error check tests
# ---------------------------------------------------------------------------


class TestNVLinkCheck:
    """Tests for NVLink error checking."""

    def test_all_active_links_pass(self) -> None:
        """VAL-RUNBOOK-006: All active links pass."""
        raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = check_nvlink_status(raw)
        assert result["passed"] is True
        assert result["inactive_links"] == []

    def test_reports_inactive_links(self) -> None:
        """VAL-RUNBOOK-006: Inactive links reported."""
        # Create a modified fixture with an inactive link
        raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        raw_modified = raw.replace(
            "Link 1\n            State                         : Active",
            "Link 1\n            State                         : Inactive",
        )
        result = check_nvlink_status(raw_modified)
        assert result["passed"] is False
        assert len(result["inactive_links"]) > 0

    def test_returns_total_link_count(self) -> None:
        raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = check_nvlink_status(raw)
        assert "total_links" in result
        assert result["total_links"] > 0


# ---------------------------------------------------------------------------
# Post-flight ECC error check tests
# ---------------------------------------------------------------------------


class TestECCCheck:
    """Tests for ECC error checking."""

    def test_zero_errors_pass(self) -> None:
        """VAL-RUNBOOK-006: Zero ECC errors pass."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = check_ecc_errors(pre_raw, post_raw)
        assert result["passed"] is True
        assert result["new_errors"] == {}

    def test_reports_new_errors(self) -> None:
        """VAL-RUNBOOK-006: New ECC errors are detected."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        # Simulate new errors by modifying post-run fixture
        post_raw_modified = post_raw.replace(
            "SRAM Correctable              : 0\n"
            "            SRAM Uncorrectable            : 0\n"
            "            DRAM Correctable              : 0\n"
            "            DRAM Uncorrectable            : 0",
            "SRAM Correctable              : 2\n"
            "            SRAM Uncorrectable            : 0\n"
            "            DRAM Correctable              : 1\n"
            "            DRAM Uncorrectable            : 0",
            1,  # replace only first occurrence (first GPU)
        )
        result = check_ecc_errors(pre_raw, post_raw_modified)
        assert result["passed"] is False
        assert len(result["new_errors"]) > 0

    def test_has_explicit_pass_fail(self) -> None:
        """VAL-RUNBOOK-006: Explicit pass/fail reporting."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = check_ecc_errors(pre_raw, post_raw)
        assert "passed" in result
        assert isinstance(result["passed"], bool)


# ---------------------------------------------------------------------------
# Full post-flight collection tests
# ---------------------------------------------------------------------------


class TestCollectPostflightChecks:
    """Tests for the full post-flight check collection."""

    def test_contains_uuid_check(self) -> None:
        """VAL-RUNBOOK-006: UUID check included."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = collect_postflight_checks(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        assert "uuid_consistency" in result
        assert result["uuid_consistency"]["passed"] is True

    def test_contains_nvlink_check(self) -> None:
        """VAL-RUNBOOK-006: NVLink check included."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = collect_postflight_checks(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        assert "nvlink_status" in result
        assert "passed" in result["nvlink_status"]

    def test_contains_ecc_check(self) -> None:
        """VAL-RUNBOOK-006: ECC check included."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = collect_postflight_checks(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        assert "ecc_errors" in result
        assert "passed" in result["ecc_errors"]

    def test_overall_pass_when_all_pass(self) -> None:
        """VAL-RUNBOOK-006: Overall pass when all checks pass."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        result = collect_postflight_checks(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        assert result["overall_passed"] is True

    def test_overall_fail_when_any_fails(self) -> None:
        """VAL-RUNBOOK-006: Overall fail when any check fails."""
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        # Drop a GPU from post to trigger UUID mismatch
        post_modified = post_raw.split("GPU 00000000:05:00.0")[0]
        result = collect_postflight_checks(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_modified,
        )
        assert result["overall_passed"] is False


# ---------------------------------------------------------------------------
# PostflightRunner tests
# ---------------------------------------------------------------------------


class TestPostflightRunner:
    """Tests for the PostflightRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        runner = PostflightRunner(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        runner = PostflightRunner(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_has_check_metrics(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        runner = PostflightRunner(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        result = runner.run()
        assert "uuid_consistency" in result.metrics
        assert "nvlink_status" in result.metrics
        assert "ecc_errors" in result.metrics
        assert "overall_passed" in result.metrics

    def test_runner_name_is_postflight(self) -> None:
        pre_raw = _load_fixture("nvidia_smi/nvidia_smi_q.txt")
        post_raw = _load_fixture("nvidia_smi/post_run_nvidia_smi_q.txt")
        runner = PostflightRunner(
            pre_nvidia_smi_q=pre_raw,
            post_nvidia_smi_q=post_raw,
        )
        result = runner.run()
        assert result.name == "post-flight"
