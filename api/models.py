"""Pydantic models for the Ornn Benchmarking API request/response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GPUInfoPayload(BaseModel):
    """GPU hardware information in the submitted run payload."""

    uuid: str = ""
    name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_mb: int = 0


class SystemInventoryPayload(BaseModel):
    """System inventory in the submitted run payload."""

    gpus: list[GPUInfoPayload] = Field(default_factory=list)
    os_info: str = ""
    kernel_version: str = ""
    cpu_model: str = ""
    numa_nodes: int = 0
    pytorch_version: str = ""


class ScorePayload(BaseModel):
    """Score data in the submitted run payload."""

    ornn_i: float | None = None
    ornn_t: float | None = None
    qualification: str | None = None
    components: dict[str, float] = Field(default_factory=dict)
    score_status: str = "error"
    score_status_detail: str | None = None
    aggregate_method: str | None = None


class SectionPayload(BaseModel):
    """Benchmark section result in the submitted run payload."""

    name: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class RunPayload(BaseModel):
    """Validated request body for ``POST /api/v1/runs``.

    Must contain the core benchmark report fields.  Server will reject
    payloads missing required fields with 422 and field-level errors.
    """

    schema_version: str = Field(
        ..., min_length=1, description="Semantic version of the report schema."
    )
    report_id: str = Field(
        ..., min_length=1, description="Client-generated unique report identifier."
    )
    created_at: str = Field(
        ..., min_length=1, description="ISO-8601 timestamp of report creation."
    )
    system_inventory: SystemInventoryPayload = Field(
        ..., description="System inventory captured during pre-flight."
    )
    sections: list[SectionPayload] = Field(
        default_factory=list, description="Benchmark section results."
    )
    scores: ScorePayload = Field(
        ..., description="Computed Ornn scores."
    )
    manifest: dict[str, object] = Field(
        default_factory=dict, description="Output manifest."
    )


class RunResponse(BaseModel):
    """Response body returned after successful run ingest."""

    run_id: str = Field(..., description="Server-assigned run identifier.")
    received_at: str = Field(..., description="Server timestamp when run was received (ISO-8601).")
    stored_at: str = Field(..., description="Server timestamp when run was persisted (ISO-8601).")
