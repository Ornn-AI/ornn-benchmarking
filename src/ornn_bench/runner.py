"""Run orchestrator for benchmark execution.

Provides the :class:`RunOrchestrator` that executes benchmark sections
in deterministic order, supports selective scope filtering, partial-failure
continuation, and progress callbacks.

Also provides :class:`DurableRunOrchestrator` for interruption-safe report
persistence that writes completed section results incrementally.
"""

from __future__ import annotations

import abc
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ornn_bench.models import (
    BenchmarkReport,
    BenchmarkStatus,
    GPUInfo,
    SectionResult,
    SystemInventory,
)

# ---------------------------------------------------------------------------
# Deterministic section execution order (VAL-RUNBOOK-007)
# ---------------------------------------------------------------------------

#: Canonical ordering: pre-flight → benchmarks → monitoring → post-flight → manifest
SECTION_ORDER: tuple[str, ...] = (
    "pre-flight",
    "compute",
    "memory",
    "interconnect",
    "monitoring",
    "post-flight",
    "manifest",
)

#: Sections that always run regardless of scope filters
INFRASTRUCTURE_SECTIONS: frozenset[str] = frozenset({
    "pre-flight",
    "monitoring",
    "post-flight",
    "manifest",
})

#: Benchmark sections that are subject to scope filtering
BENCHMARK_SECTIONS: frozenset[str] = frozenset({
    "compute",
    "memory",
    "interconnect",
})

# Type alias for progress callbacks
ProgressCallback = Callable[[str, str], None]


def _as_dict(value: object) -> dict[str, object]:
    """Return a shallow string-keyed dict when possible."""
    if not isinstance(value, dict):
        return {}
    return {str(key): val for key, val in value.items()}


def _as_list_of_dicts(value: object) -> list[dict[str, object]]:
    """Return a list containing only dict items."""
    if not isinstance(value, list):
        return []

    items: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            items.append({str(key): val for key, val in item.items()})
    return items


def _as_str(value: object) -> str:
    """Coerce a value to string when it is already textual."""
    return value if isinstance(value, str) else ""


def _as_int(value: object) -> int:
    """Coerce an integer-like value to int."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return 0
    return 0


def _find_section(sections: list[SectionResult], name: str) -> SectionResult | None:
    """Return the section result with the matching name, if present."""
    for section in sections:
        if section.name == name:
            return section
    return None


def _derive_system_inventory(sections: list[SectionResult]) -> SystemInventory:
    """Build top-level system inventory from pre-flight metrics."""
    preflight = _find_section(sections, "pre-flight")
    if preflight is None or preflight.status != BenchmarkStatus.COMPLETED:
        return SystemInventory()

    metrics = _as_dict(preflight.metrics)
    software_versions = _as_dict(metrics.get("software_versions"))
    gpu_inventory = _as_dict(metrics.get("gpu_inventory"))
    gpu_entries = _as_list_of_dicts(gpu_inventory.get("gpus"))
    if not gpu_entries:
        gpu_entries = _as_list_of_dicts(metrics.get("gpus"))

    driver_version = _as_str(
        metrics.get("driver_version") or software_versions.get("driver")
    )
    cuda_version = _as_str(
        metrics.get("cuda_version") or software_versions.get("cuda")
    )

    gpus: list[GPUInfo] = []
    for gpu in gpu_entries:
        gpus.append(
            GPUInfo(
                uuid=_as_str(gpu.get("uuid")),
                name=_as_str(gpu.get("name") or gpu.get("product_name")),
                driver_version=_as_str(gpu.get("driver_version") or driver_version),
                cuda_version=_as_str(gpu.get("cuda_version") or cuda_version),
                memory_total_mb=_as_int(
                    gpu.get("memory_total_mb") or gpu.get("memory_total_mib")
                ),
            )
        )

    return SystemInventory(
        gpus=gpus,
        os_info=_as_str(metrics.get("os_info") or metrics.get("os")),
        kernel_version=_as_str(metrics.get("kernel_version") or metrics.get("kernel")),
        cpu_model=_as_str(metrics.get("cpu_model")),
        numa_nodes=_as_int(metrics.get("numa_nodes")),
        pytorch_version=_as_str(metrics.get("pytorch_version")),
    )


def _derive_manifest(sections: list[SectionResult]) -> dict[str, object]:
    """Build top-level manifest from the manifest section metrics."""
    manifest = _find_section(sections, "manifest")
    if manifest is None or manifest.status != BenchmarkStatus.COMPLETED:
        return {}
    return _as_dict(manifest.metrics)


def _build_benchmark_report(
    *,
    report_id: str,
    created_at: str,
    sections: list[SectionResult],
) -> BenchmarkReport:
    """Construct the benchmark report with derived top-level sections."""
    from ornn_bench.scoring import derive_scores_from_sections

    return BenchmarkReport(
        schema_version="1.0.0",
        report_id=report_id,
        created_at=created_at,
        system_inventory=_derive_system_inventory(sections),
        sections=sections,
        scores=derive_scores_from_sections(sections),
        manifest=_derive_manifest(sections),
    )


# ---------------------------------------------------------------------------
# Section runner interface
# ---------------------------------------------------------------------------


class SectionRunner(abc.ABC):
    """Abstract base class for a benchmark section runner.

    Each concrete runner implements :meth:`run` which performs the actual
    benchmark work and returns a :class:`SectionResult`.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def run(self) -> SectionResult:
        """Execute the section and return a result.

        Implementations should set ``started_at`` / ``finished_at``
        timestamps on the returned :class:`SectionResult`.
        """


# ---------------------------------------------------------------------------
# Stub runners (used when benchmark tooling is not yet implemented)
# ---------------------------------------------------------------------------


class StubSectionRunner(SectionRunner):
    """Placeholder runner that immediately succeeds with empty metrics.

    Used during development before real benchmark integrations are wired up.
    """

    def run(self) -> SectionResult:
        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            metrics={},
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RunOrchestrator:
    """Orchestrates benchmark section execution.

    Parameters
    ----------
    runners:
        Mapping of section name → :class:`SectionRunner` instances.
        Must include all names from :data:`SECTION_ORDER`.
    scope:
        Optional set of benchmark section names to run (e.g. ``{"compute"}``).
        When provided, only listed benchmark sections execute; infrastructure
        sections (pre-flight, post-flight, manifest) always run.
        When ``None`` (default), all sections execute.
    on_progress:
        Optional callback ``(section_name, status_string) -> None`` invoked
        when a section starts or finishes.
    """

    def __init__(
        self,
        runners: dict[str, SectionRunner],
        scope: set[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._runners = runners
        self._scope = scope
        self._on_progress = on_progress
        self._results: list[SectionResult] = []
        self._has_failures = False

    @property
    def has_failures(self) -> bool:
        """Return True if any section failed or timed out."""
        return self._has_failures

    def _should_run(self, section_name: str) -> bool:
        """Determine whether a section should run given the current scope."""
        # Infrastructure sections always run
        if section_name in INFRASTRUCTURE_SECTIONS:
            return True
        # If no scope filter, run everything
        if self._scope is None:
            return True
        # Only run if section is in scope
        return section_name in self._scope

    def _emit_progress(self, section: str, status: str) -> None:
        """Invoke progress callback if set."""
        if self._on_progress is not None:
            self._on_progress(section, status)

    def execute(self) -> BenchmarkReport:
        """Execute all sections in deterministic order.

        Returns a :class:`BenchmarkReport` with results for every section.
        Sections that fail or raise exceptions are recorded with appropriate
        status but do not abort remaining sections.
        """
        self._results = []
        self._has_failures = False

        for section_name in SECTION_ORDER:
            if not self._should_run(section_name):
                # Mark as skipped
                self._emit_progress(section_name, "skipped")
                self._results.append(
                    SectionResult(
                        name=section_name,
                        status=BenchmarkStatus.SKIPPED,
                    )
                )
                continue

            runner = self._runners.get(section_name)
            if runner is None:
                # No runner registered for this section
                self._emit_progress(section_name, "skipped")
                self._results.append(
                    SectionResult(
                        name=section_name,
                        status=BenchmarkStatus.SKIPPED,
                        error=f"No runner registered for section '{section_name}'",
                    )
                )
                continue

            # If this is the manifest runner, provide it the accumulated sections
            if section_name == "manifest" and hasattr(runner, "set_sections"):
                runner.set_sections(list(self._results))

            self._emit_progress(section_name, "started")

            try:
                result = runner.run()
                # Ensure the result has the correct name
                result.name = section_name
            except Exception as exc:
                # Catch unexpected exceptions — record as FAILED
                now = datetime.now(timezone.utc).isoformat()
                result = SectionResult(
                    name=section_name,
                    status=BenchmarkStatus.FAILED,
                    started_at=now,
                    finished_at=now,
                    error=f"Unexpected error: {exc}",
                )

            self._results.append(result)

            # Track failures
            if result.status in (BenchmarkStatus.FAILED, BenchmarkStatus.TIMEOUT):
                self._has_failures = True
                status_str = result.status.value
            else:
                status_str = "completed"

            self._emit_progress(section_name, status_str)

        return self._build_report()

    def _build_report(self) -> BenchmarkReport:
        """Build a BenchmarkReport from collected section results."""
        now = datetime.now(timezone.utc).isoformat()
        return _build_benchmark_report(
            report_id=str(uuid.uuid4()),
            created_at=now,
            sections=self._results,
        )


# ---------------------------------------------------------------------------
# Durable orchestrator (VAL-RUNBOOK-009)
# ---------------------------------------------------------------------------


class DurableRunOrchestrator(RunOrchestrator):
    """Orchestrator that persists report incrementally after each section.

    Extends :class:`RunOrchestrator` to write the report to disk after
    every section completes, ensuring that interrupted or failing runs
    still have completed section outputs persisted.

    Parameters
    ----------
    runners:
        Mapping of section name → SectionRunner.
    output_path:
        Path where the JSON report is written incrementally.
    scope:
        Optional set of benchmark section names to run.
    on_progress:
        Optional progress callback.
    """

    def __init__(
        self,
        runners: dict[str, SectionRunner],
        output_path: Path,
        scope: set[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        super().__init__(runners=runners, scope=scope, on_progress=on_progress)
        self._output_path = output_path
        self._report_id = str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc).isoformat()

    def _persist_current_state(self) -> None:
        """Write current report state to disk."""
        report = _build_benchmark_report(
            report_id=self._report_id,
            created_at=self._created_at,
            sections=list(self._results),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(report.model_dump_json(indent=2))

    def execute(self) -> BenchmarkReport:
        """Execute all sections with incremental persistence.

        Each section's result is written to disk immediately after
        completion. If a KeyboardInterrupt occurs, the current state
        is persisted before returning.
        """
        self._results = []
        self._has_failures = False
        interrupted = False

        for section_name in SECTION_ORDER:
            if interrupted:
                # Mark remaining sections as skipped after interruption
                self._results.append(
                    SectionResult(
                        name=section_name,
                        status=BenchmarkStatus.SKIPPED,
                        error="Run interrupted before this section",
                    )
                )
                continue

            if not self._should_run(section_name):
                self._emit_progress(section_name, "skipped")
                self._results.append(
                    SectionResult(
                        name=section_name,
                        status=BenchmarkStatus.SKIPPED,
                    )
                )
                self._persist_current_state()
                continue

            runner = self._runners.get(section_name)
            if runner is None:
                self._emit_progress(section_name, "skipped")
                self._results.append(
                    SectionResult(
                        name=section_name,
                        status=BenchmarkStatus.SKIPPED,
                        error=f"No runner registered for section '{section_name}'",
                    )
                )
                self._persist_current_state()
                continue

            # If this is the manifest runner, provide it the accumulated sections
            if section_name == "manifest" and hasattr(runner, "set_sections"):
                runner.set_sections(list(self._results))

            self._emit_progress(section_name, "started")

            try:
                result = runner.run()
                result.name = section_name
            except KeyboardInterrupt:
                now = datetime.now(timezone.utc).isoformat()
                result = SectionResult(
                    name=section_name,
                    status=BenchmarkStatus.FAILED,
                    started_at=now,
                    finished_at=now,
                    error="Run interrupted by user",
                )
                interrupted = True
            except Exception as exc:
                now = datetime.now(timezone.utc).isoformat()
                result = SectionResult(
                    name=section_name,
                    status=BenchmarkStatus.FAILED,
                    started_at=now,
                    finished_at=now,
                    error=f"Unexpected error: {exc}",
                )

            self._results.append(result)

            if result.status in (BenchmarkStatus.FAILED, BenchmarkStatus.TIMEOUT):
                self._has_failures = True
                status_str = result.status.value
            else:
                status_str = "completed"

            self._emit_progress(section_name, status_str)

            # Persist after each section
            self._persist_current_state()

        return self._build_report()

    def _build_report(self) -> BenchmarkReport:
        """Build a BenchmarkReport using the pre-assigned report ID."""
        return _build_benchmark_report(
            report_id=self._report_id,
            created_at=self._created_at,
            sections=self._results,
        )


# ---------------------------------------------------------------------------
# Factory: build default section runners
# ---------------------------------------------------------------------------


def build_section_runners() -> dict[str, SectionRunner]:
    """Build the default set of section runners.

    Uses real runbook runners for all sections: pre-flight, compute,
    memory, interconnect, monitoring, post-flight, and manifest.
    """
    from ornn_bench.runbook.compute import ComputeMatrixRunner
    from ornn_bench.runbook.interconnect import InterconnectMatrixRunner
    from ornn_bench.runbook.manifest import ManifestRunner
    from ornn_bench.runbook.memory import MemoryMatrixRunner
    from ornn_bench.runbook.monitoring import MonitoringRunner
    from ornn_bench.runbook.postflight import PostflightRunner
    from ornn_bench.runbook.preflight import PreflightRunner

    runners: dict[str, SectionRunner] = {
        "pre-flight": PreflightRunner(),
        "compute": ComputeMatrixRunner(),
        "memory": MemoryMatrixRunner(),
        "interconnect": InterconnectMatrixRunner(),
        "monitoring": MonitoringRunner(),
        "post-flight": PostflightRunner(pre_nvidia_smi_q=""),
        "manifest": ManifestRunner(),
    }
    return runners
