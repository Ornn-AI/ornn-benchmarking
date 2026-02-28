# Ornn GPU Benchmarking CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A standardized GPU benchmarking framework that runs 30+ benchmarks across compute, memory, and interconnect categories, computes **Ornn-I** (Inference) and **Ornn-T** (Training) composite scores, and qualifies GPUs as **Premium**, **Standard**, or **Below** grade.

## Features

- **Full Section 8 Runbook** — Pre-flight inventory, compute (MAMF), memory (nvbandwidth), interconnect (NCCL), thermal monitoring, and post-flight checks
- **Composite Scoring** — Ornn-I and Ornn-T scores with deterministic qualification grading
- **Beautiful Terminal Output** — Rich-powered progress bars, colored scorecards, and qualification badges
- **JSON Reports** — Machine-readable reports with complete benchmark data and manifest
- **Cloud Upload** — Submit results to the Ornn API with idempotent retry-safe uploads
- **Score Verification** — Server-side recomputation to validate local scores
- **Multi-GPU Support** — Per-GPU isolation, scoring, and minimum-based aggregation

## Quick Start

### Install

```bash
pip install ornn-bench
```

### Run Benchmarks

```bash
# Full benchmark suite
ornn-bench run

# Compute benchmarks only
ornn-bench run --compute-only

# Memory benchmarks only
ornn-bench run --memory-only

# Run and upload results
ornn-bench run --upload --api-key YOUR_KEY
```

### Check Your Environment

```bash
ornn-bench info
```

### View a Saved Report

```bash
ornn-bench report ornn_report_abc12345.json
```

### Upload a Report

```bash
ornn-bench upload ornn_report_abc12345.json --api-key YOUR_KEY
```

## Scoring

The scoring engine computes two composite scores:

| Score | Formula | Components |
|-------|---------|------------|
| **Ornn-I** (Inference) | `55 × (BW / BW_ref) + 45 × (FP8 / FP8_ref)` | Memory bandwidth + FP8 compute |
| **Ornn-T** (Training) | `55 × (BF16 / BF16_ref) + 45 × (AR / AR_ref)` | BF16 compute + all-reduce bandwidth |

### Qualification Grades

| Grade | Composite Gate | Floor Checks |
|-------|---------------|--------------|
| **Premium** | ≥ 90 | Ornn-I ≥ 80 AND Ornn-T ≥ 80 |
| **Standard** | ≥ 70 | Ornn-I ≥ 60 AND Ornn-T ≥ 60 |
| **Below** | < 70 | — |

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with drivers installed (`nvidia-smi` on PATH)
- **CUDA Toolkit** compatible with your GPU
- **Benchmark tools**: `nvbandwidth`, `nccl-tests`, `mamf-finder.py` (see [CLI Usage Guide](docs/cli-usage.md) for details)

> **Note:** `ornn-bench info` and `ornn-bench report` work on machines without GPU hardware.

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Usage Guide](docs/cli-usage.md) | All commands, flags, output formats, and troubleshooting |
| [API Reference](docs/api-reference.md) | Endpoints, schemas, authentication, error codes, rate limits |
| [Deployment Guide](docs/deployment.md) | GCP setup, environment variables, deploy commands, verification |
| [Architecture](docs/architecture.md) | System components, data flow, scoring pipeline, deployment topology |
| [Contributing](CONTRIBUTING.md) | Development setup, code style, testing, and PR workflow |

## Development

```bash
# Clone and install
git clone https://github.com/Ornn-AI/ornn-benchmarking.git
cd ornn-benchmarking
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest -q

# Lint
ruff check .

# Type check
mypy src api
```

## License

This project is licensed under the MIT License. See [pyproject.toml](pyproject.toml) for details.
