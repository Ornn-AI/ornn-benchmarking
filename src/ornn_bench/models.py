"""Pydantic data models for benchmark results and reports."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BenchmarkStatus(str, Enum):
    """Status of a benchmark section execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class Qualification(str, Enum):
    """GPU qualification grade."""

    PREMIUM = "Premium"
    STANDARD = "Standard"
    BELOW = "Below"


class GPUInfo(BaseModel):
    """GPU hardware information."""

    uuid: str = ""
    name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_mb: int = 0


class SystemInventory(BaseModel):
    """System inventory captured during pre-flight."""

    gpus: list[GPUInfo] = Field(default_factory=list)
    os_info: str = ""
    kernel_version: str = ""
    cpu_model: str = ""
    numa_nodes: int = 0
    pytorch_version: str = ""


class ScoreResult(BaseModel):
    """Computed Ornn scores."""

    ornn_i: float | None = None
    ornn_t: float | None = None
    qualification: Qualification | None = None
    components: dict[str, float] = Field(default_factory=dict)


class SectionResult(BaseModel):
    """Result of a single benchmark section."""

    name: str
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class BenchmarkReport(BaseModel):
    """Complete benchmark report."""

    schema_version: str = "1.0.0"
    report_id: str = ""
    created_at: str = ""
    system_inventory: SystemInventory = Field(default_factory=SystemInventory)
    sections: list[SectionResult] = Field(default_factory=list)
    scores: ScoreResult = Field(default_factory=ScoreResult)
    manifest: dict[str, object] = Field(default_factory=dict)
