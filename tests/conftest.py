"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ornn_bench.models import (
    BenchmarkReport,
    BenchmarkStatus,
    GPUInfo,
    ScoreResult,
    SectionResult,
    SystemInventory,
)


@pytest.fixture()
def sample_gpu_info() -> GPUInfo:
    """Return a sample GPUInfo for testing."""
    return GPUInfo(
        uuid="GPU-12345678-abcd-1234-abcd-123456789abc",
        name="NVIDIA H100 80GB HBM3",
        driver_version="535.129.03",
        cuda_version="12.2",
        memory_total_mb=81920,
    )


@pytest.fixture()
def sample_system_inventory(sample_gpu_info: GPUInfo) -> SystemInventory:
    """Return a sample SystemInventory for testing."""
    return SystemInventory(
        gpus=[sample_gpu_info],
        os_info="Ubuntu 22.04.3 LTS",
        kernel_version="5.15.0-91-generic",
        cpu_model="Intel(R) Xeon(R) Platinum 8480+",
        numa_nodes=2,
        pytorch_version="2.1.2",
    )


@pytest.fixture()
def sample_report(sample_system_inventory: SystemInventory) -> BenchmarkReport:
    """Return a sample BenchmarkReport for testing."""
    return BenchmarkReport(
        schema_version="1.0.0",
        report_id="test-run-001",
        created_at="2024-01-15T10:30:00Z",
        system_inventory=sample_system_inventory,
        sections=[
            SectionResult(
                name="compute",
                status=BenchmarkStatus.COMPLETED,
                started_at="2024-01-15T10:30:01Z",
                finished_at="2024-01-15T10:35:00Z",
            ),
            SectionResult(
                name="memory",
                status=BenchmarkStatus.COMPLETED,
                started_at="2024-01-15T10:35:01Z",
                finished_at="2024-01-15T10:40:00Z",
            ),
        ],
        scores=ScoreResult(
            ornn_i=85.5,
            ornn_t=78.2,
            qualification=None,
            components={"bw": 1.2, "fp8": 0.9, "bf16": 1.1, "ar": 0.8},
        ),
    )
