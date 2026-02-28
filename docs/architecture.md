# Architecture

This document describes the system components, data flow, scoring pipeline, and deployment topology of the Ornn GPU Benchmarking Framework.

---

## System Overview

The framework consists of three main components:

```
┌─────────────────────┐         ┌──────────────────────┐         ┌──────────────┐
│   ornn-bench CLI    │──HTTP──▶│   FastAPI Backend    │──gRPC──▶│  Firestore   │
│   (Python package)  │         │   (Cloud Run)        │         │  (GCP)       │
└─────────────────────┘         └──────────────────────┘         └──────────────┘
         │                                │
         │ Local JSON                     │ Score verification
         ▼                                ▼
   ornn_report_*.json              Server recomputation
```

### 1. CLI (`ornn-bench`)

The Python CLI package installed via `pip install ornn-bench`. Runs on NVIDIA GPU machines to execute benchmarks, compute scores locally, and optionally upload results.

**Location:** `src/ornn_bench/`

| Module | Responsibility |
|--------|---------------|
| `cli.py` | Typer-based command hierarchy (`run`, `info`, `report`, `upload`) |
| `runner.py` | Run orchestrator — deterministic section execution, scope filtering, partial failure |
| `system.py` | System/GPU probing via `nvidia-smi`, tool detection, environment diagnostics |
| `scoring.py` | Ornn-I/Ornn-T computation, qualification grading, multi-GPU aggregation |
| `display.py` | Rich terminal scorecard, plain text, and JSON output rendering |
| `models.py` | Pydantic data models for reports, sections, scores |
| `api_client.py` | HTTP client for API upload, verification, schema version checks |
| `runbook/` | Section runners (compute, memory, interconnect, monitoring, etc.) |

### 2. FastAPI Backend

A stateless API service deployed on Google Cloud Run. Receives benchmark reports, stores them in Firestore, and provides server-side score verification.

**Location:** `api/`

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI application factory |
| `config.py` | Settings from environment variables |
| `auth.py` | API key validation (valid + revoked key management) |
| `rate_limit.py` | Per-key sliding-window rate limiter |
| `dependencies.py` | Dependency injection (Firestore client, settings, rate limiter) |
| `models.py` | Pydantic request/response schemas |
| `scoring.py` | Server-side score recomputation for verification |
| `routers/health.py` | `GET /health` liveness probe |
| `routers/runs.py` | `POST /api/v1/runs` and `GET /api/v1/runs/{id}` |
| `routers/verify.py` | `POST /api/v1/verify` score verification |

### 3. Firestore

Google Cloud Firestore in native mode stores benchmark run documents. No SQL database, no always-on infrastructure.

**Collection:** `benchmark_runs`

**Document structure:**
- Server metadata: `run_id`, `received_at`, `stored_at`, `dedupe_key`
- Client payload: `schema_version`, `report_id`, `created_at`, `system_inventory`, `sections`, `scores`, `manifest`

---

## Data Flow

### Benchmark Execution Flow

```
ornn-bench run
    │
    ├─▶ check_gpu_available()        # Verify GPU presence
    ├─▶ build_section_runners()      # Create runbook section runners
    ├─▶ RunOrchestrator.execute()    # Execute sections in order
    │       │
    │       ├─▶ pre-flight           # System inventory, UUID collection
    │       ├─▶ compute              # MAMF benchmarks (BF16, FP8, FP16, TF32)
    │       ├─▶ memory               # nvbandwidth tests (7 types)
    │       ├─▶ interconnect         # NCCL tests (6 types)
    │       ├─▶ monitoring           # Thermal/power snapshots
    │       ├─▶ post-flight          # UUID consistency, error checks
    │       └─▶ manifest             # Output artifact inventory
    │
    ├─▶ compute_scores()             # Local Ornn-I/Ornn-T computation
    ├─▶ render_scorecard()           # Terminal display
    ├─▶ write JSON report            # Persist to disk
    │
    └─▶ (if --upload)
            ├─▶ api_client.upload()  # POST /api/v1/runs
            └─▶ api_client.verify()  # POST /api/v1/verify
```

### Upload Flow

```
CLI                              API                         Firestore
 │                                │                              │
 │  POST /api/v1/runs             │                              │
 │  X-API-Key: xxx                │                              │
 │  { report payload }            │                              │
 │──────────────────────────────▶│                              │
 │                                │ validate API key             │
 │                                │ check rate limit             │
 │                                │ check schema version         │
 │                                │ compute dedupe key           │
 │                                │ check for existing doc ─────▶│
 │                                │                    ◀─────────│
 │                                │ (if new) store doc ─────────▶│
 │                                │                    ◀─────────│
 │  201 { run_id, timestamps }    │                              │
 │◀──────────────────────────────│                              │
 │                                │                              │
 │  POST /api/v1/verify           │                              │
 │  { components, scores }        │                              │
 │──────────────────────────────▶│                              │
 │                                │ recompute Ornn-I/T           │
 │                                │ compare with submitted       │
 │  200 { status, details }       │                              │
 │◀──────────────────────────────│                              │
```

---

## Scoring Pipeline

### Score Computation

Scores are computed identically on both client and server:

```
Input Metrics          Score Formulas                    Qualification
─────────────          ──────────────                    ─────────────
BW   (bandwidth)  ──▶  Ornn-I = 55×(BW/BW_ref)         ┌─── Premium ◀── composite ≥ 90
FP8  (compute)    ──▶         + 45×(FP8/FP8_ref)        │                AND both ≥ 80
                                                         │
BF16 (compute)    ──▶  Ornn-T = 55×(BF16/BF16_ref)      ├─── Standard ◀── composite ≥ 70
AR   (all-reduce) ──▶         + 45×(AR/AR_ref)           │                 AND both ≥ 60
                                                         │
                       Composite = (Ornn-I + Ornn-T) / 2 └─── Below ◀── otherwise
```

### Multi-GPU Aggregation

For systems with multiple GPUs:

1. Each GPU is benchmarked in isolation (`CUDA_VISIBLE_DEVICES`)
2. Per-GPU Ornn-I and Ornn-T are computed independently
3. Final scores use **minimum** aggregation — the weakest GPU gates qualification
4. Per-GPU breakdowns are recorded in the report for transparency

### Score Status

Scoring uses explicit status tracking rather than silent NaN:

| Status | Meaning |
|--------|---------|
| `valid` | Both Ornn-I and Ornn-T computed successfully |
| `partial` | One score computed; the other has missing/invalid metrics |
| `error` | No valid metrics available for scoring |

### Edge Case Handling

- **NaN/Inf inputs** → Rejected, score returns `None`
- **Negative values** → Rejected as invalid
- **Zero reference values** → Rejected (division by zero prevention)
- **Missing metrics** → Explicit `partial`/`error` status with detail message

---

## Benchmark Execution Model

### Section Runners

Each benchmark section is implemented as a runner class following a common interface:

```python
class SectionRunner(ABC):
    @abstractmethod
    def run(self) -> SectionResult:
        """Execute the benchmark section and return results."""
```

Runners wrap external GPU tools via subprocess:

| Section | External Tool | Key Metrics |
|---------|--------------|-------------|
| Compute | `mamf-finder.py` | TFLOPS per dtype (BF16, FP8, FP16, TF32) |
| Memory | `nvbandwidth` | Bandwidth per test type (GB/s) |
| Interconnect | NCCL tests (`all_reduce_perf`, etc.) | Bus bandwidth (GB/s) |

### Scope Filtering

The `--compute-only`, `--memory-only`, and `--interconnect-only` flags filter which benchmark sections run. Infrastructure sections (pre-flight, monitoring, post-flight, manifest) always execute.

### Partial Failure Semantics

The orchestrator uses a fail-continue strategy:

1. Each section runs in a try-catch boundary
2. Failures are recorded with status `failed` or `timeout`
3. Remaining sections continue execution
4. The report captures all results, including failures
5. Scoring handles missing metrics with explicit `partial`/`error` status

---

## Deployment Topology

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                  │
│                    Project: ornn-benchmarking             │
│                                                          │
│  ┌──────────────────────┐     ┌───────────────────────┐ │
│  │     Cloud Run         │     │      Firestore        │ │
│  │     (us-east1)        │     │      (native mode)    │ │
│  │                       │     │      (us-east1)       │ │
│  │  ┌─────────────────┐ │     │                       │ │
│  │  │   FastAPI App    │─┼────▶│  benchmark_runs      │ │
│  │  │   (ornn-api)     │ │     │  collection          │ │
│  │  └─────────────────┘ │     │                       │ │
│  │                       │     └───────────────────────┘ │
│  │  Min instances: 0     │                               │
│  │  Max instances: 2     │                               │
│  │  Memory: 256Mi        │                               │
│  │  CPU: 1 vCPU          │                               │
│  └──────────────────────┘                                │
│                                                          │
│  Cost: $0 at low usage (free tier)                       │
└─────────────────────────────────────────────────────────┘
              ▲
              │ HTTPS
              │
┌─────────────┴─────────────────┐
│  GPU Machine (user's hardware) │
│                                │
│  pip install ornn-bench        │
│  ornn-bench run --upload       │
└────────────────────────────────┘
```

### Free-Tier-Safe Design

The architecture deliberately avoids always-on paid infrastructure:

| Component | Cost Model |
|-----------|-----------|
| Cloud Run | Scale-to-zero, pay-per-request. Free tier: 2M requests/month |
| Firestore | Pay-per-operation. Free tier: 20K writes/day, 50K reads/day |

**No VMs, Cloud SQL, GKE, or persistent compute is used.**

---

## Project Structure

```
ornn-benchmarking/
├── src/ornn_bench/           # CLI package source
│   ├── __init__.py           # Package version
│   ├── cli.py                # Command definitions (run, info, report, upload)
│   ├── runner.py             # Benchmark orchestrator
│   ├── system.py             # GPU/tool/environment probing
│   ├── scoring.py            # Ornn-I/T scoring engine
│   ├── display.py            # Terminal output (Rich + plain text)
│   ├── models.py             # Pydantic data models
│   ├── api_client.py         # API HTTP client
│   └── runbook/              # Section runner implementations
│       ├── preflight.py      # Pre-flight system inventory
│       ├── compute.py        # Compute benchmark runner
│       ├── memory.py         # Memory benchmark runner
│       ├── interconnect.py   # Interconnect benchmark runner
│       ├── monitoring.py     # Thermal/power monitoring
│       ├── postflight.py     # Post-flight consistency checks
│       ├── manifest.py       # Output manifest generation
│       └── parsers.py        # Tool output parsers
├── api/                      # FastAPI backend source
│   ├── main.py               # App factory
│   ├── config.py             # Environment-based settings
│   ├── auth.py               # API key authentication
│   ├── rate_limit.py         # Sliding-window rate limiter
│   ├── dependencies.py       # Dependency injection
│   ├── models.py             # Request/response schemas
│   ├── scoring.py            # Server-side score recomputation
│   └── routers/              # API endpoint handlers
├── tests/                    # Test suite
│   ├── api/                  # API endpoint tests
│   ├── cli/                  # CLI command tests
│   ├── runbook/              # Runbook section tests
│   ├── scoring/              # Scoring engine tests
│   ├── fixtures/             # Sample benchmark tool outputs
│   └── integration/          # End-to-end integration tests
├── docs/                     # Documentation
├── deploy.sh                 # Cloud Run deployment script
├── Dockerfile                # API container image
└── pyproject.toml            # Package metadata and tool config
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| CLI Framework | [Typer](https://typer.tiangolo.com/) |
| Terminal UI | [Rich](https://rich.readthedocs.io/) |
| Data Models | [Pydantic v2](https://docs.pydantic.dev/) |
| HTTP Client | [httpx](https://www.python-httpx.org/) |
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) |
| Database | [Google Cloud Firestore](https://cloud.google.com/firestore) |
| Testing | [pytest](https://docs.pytest.org/) |
| Linting | [Ruff](https://docs.astral.sh/ruff/) |
| Type Checking | [mypy](https://mypy.readthedocs.io/) |
| Packaging | [setuptools](https://setuptools.pypa.io/) (PEP 621) |
| Deployment | [Google Cloud Run](https://cloud.google.com/run) |
