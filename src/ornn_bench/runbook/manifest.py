"""Manifest builder for tracking produced, skipped, and missing artifacts.

The manifest explicitly records which artifacts were produced by each
runbook section, which were skipped (with reason), and which are missing
due to failures (with failure reason). This satisfies VAL-RUNBOOK-008.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runner import SectionRunner


class ManifestStatus(str, Enum):
    """Status of a manifest entry."""

    PRODUCED = "produced"
    SKIPPED = "skipped"
    MISSING = "missing"


class ManifestEntry:
    """A single manifest entry tracking an artifact."""

    def __init__(
        self,
        section: str,
        artifact: str,
        status: ManifestStatus,
        reason: str = "",
    ) -> None:
        self.section = section
        self.artifact = artifact
        self.status = status
        self.reason = reason

    def to_dict(self) -> dict[str, str]:
        """Serialize to dict."""
        d: dict[str, str] = {
            "section": self.section,
            "artifact": self.artifact,
            "status": self.status.value,
        }
        if self.reason:
            d["reason"] = self.reason
        return d


#: Expected artifacts per section
SECTION_ARTIFACTS: dict[str, list[str]] = {
    "pre-flight": [
        "gpu_inventory",
        "nvlink_topology",
        "ecc_baseline",
        "software_versions",
        "system_info",
    ],
    "compute": [
        "bf16_results",
        "fp8_e4m3_results",
        "fp8_e5m2_results",
        "fp16_results",
        "fixed_shape_results",
    ],
    "memory": [
        "nvbandwidth_results",
        "pytorch_d2d_crossval",
    ],
    "interconnect": [
        "nccl_results",
        "bus_bandwidth_summary",
    ],
    "monitoring": [
        "pre_snapshot",
        "post_snapshot",
        "time_series",
        "xid_errors",
    ],
    "post-flight": [
        "uuid_consistency",
        "nvlink_status",
        "ecc_errors",
    ],
}


class ManifestBuilder:
    """Builds an artifact manifest from section results.

    Tracks which artifacts were produced, skipped, or missing with
    explicit reasons for each omission.
    """

    def __init__(self) -> None:
        self._entries: list[ManifestEntry] = []

    def record_produced(self, section: str, artifact: str) -> None:
        """Record a produced artifact."""
        self._entries.append(
            ManifestEntry(section, artifact, ManifestStatus.PRODUCED)
        )

    def record_skipped(
        self, section: str, artifact: str, *, reason: str = ""
    ) -> None:
        """Record a skipped artifact with reason."""
        self._entries.append(
            ManifestEntry(section, artifact, ManifestStatus.SKIPPED, reason)
        )

    def record_missing(
        self, section: str, artifact: str, *, reason: str = ""
    ) -> None:
        """Record a missing artifact with failure reason."""
        self._entries.append(
            ManifestEntry(section, artifact, ManifestStatus.MISSING, reason)
        )

    def get_entries(self) -> list[ManifestEntry]:
        """Return all manifest entries."""
        return list(self._entries)

    def build_from_sections(self, sections: list[SectionResult]) -> None:
        """Build manifest entries from section results.

        For each section, checks expected artifacts and records them as
        produced (if section completed), skipped (if section skipped),
        or missing (if section failed).
        """
        for section in sections:
            expected_artifacts = SECTION_ARTIFACTS.get(section.name, [])

            if section.status == BenchmarkStatus.COMPLETED:
                # Record each expected artifact as produced
                for artifact in expected_artifacts:
                    self.record_produced(section.name, artifact)
            elif section.status == BenchmarkStatus.SKIPPED:
                # Record each expected artifact as skipped
                reason = section.error or "Section was skipped"
                for artifact in expected_artifacts:
                    self.record_skipped(section.name, artifact, reason=reason)
            elif section.status in (
                BenchmarkStatus.FAILED,
                BenchmarkStatus.TIMEOUT,
            ):
                # Record each expected artifact as missing
                reason = section.error or f"Section {section.status.value}"
                for artifact in expected_artifacts:
                    self.record_missing(section.name, artifact, reason=reason)
            elif section.status == BenchmarkStatus.PENDING:
                # Section never ran
                for artifact in expected_artifacts:
                    self.record_skipped(
                        section.name, artifact, reason="Section did not execute"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a dict suitable for JSON."""
        entries_list = [e.to_dict() for e in self._entries]

        produced = sum(1 for e in self._entries if e.status == ManifestStatus.PRODUCED)
        skipped = sum(1 for e in self._entries if e.status == ManifestStatus.SKIPPED)
        missing = sum(1 for e in self._entries if e.status == ManifestStatus.MISSING)

        return {
            "entries": entries_list,
            "summary": {
                "total": len(self._entries),
                "produced": produced,
                "skipped": skipped,
                "missing": missing,
            },
        }


class ManifestRunner(SectionRunner):
    """Section runner that builds the artifact manifest.

    Takes the list of all other section results and produces a manifest
    recording produced, skipped, and missing artifacts.

    Parameters
    ----------
    sections:
        List of section results to build manifest from.
    """

    def __init__(self, *, sections: list[SectionResult] | None = None) -> None:
        super().__init__("manifest")
        self._sections = sections or []

    def set_sections(self, sections: list[SectionResult]) -> None:
        """Update sections to build manifest from (used by orchestrator)."""
        self._sections = sections

    def run(self) -> SectionResult:
        """Execute manifest building."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            builder = ManifestBuilder()
            builder.build_from_sections(self._sections)
            manifest = builder.to_dict()

            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                metrics=manifest,
            )
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                error=f"Manifest build failed: {exc}",
            )
