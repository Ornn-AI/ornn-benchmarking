# CLI Usage Guide

The `ornn-bench` CLI runs standardized GPU benchmarks, computes Ornn-I and Ornn-T scores, and optionally uploads results to the Ornn API.

## Installation

```bash
pip install ornn-bench
```

Verify the installation:

```bash
ornn-bench --version
# ornn-bench 0.1.0
```

## Commands

### `ornn-bench --help`

```
Usage: ornn-bench [OPTIONS] COMMAND [ARGS]...

  Ornn GPU Benchmarking CLI — run standardized GPU benchmarks and compute scores.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  info    Display system and GPU environment information.
  report  Re-render a previously saved benchmark report.
  run     Run the full GPU benchmark suite (or selected sections).
  upload  Upload a benchmark report to the Ornn API.
```

---

### `ornn-bench run`

Run the full GPU benchmark suite or selected sections.

```bash
ornn-bench run [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--compute-only` | Run only compute benchmarks |
| `--memory-only` | Run only memory benchmarks |
| `--interconnect-only` | Run only interconnect benchmarks |
| `--output`, `-o PATH` | Path for the JSON report file (default: `ornn_report_<id>.json`) |
| `--upload` | Upload results to the Ornn API after the run |
| `--api-key TEXT` | API key for upload (also via `ORNN_API_KEY` env var) |

**Examples:**

```bash
# Full benchmark suite
ornn-bench run

# Save report to a specific path
ornn-bench run -o results/my_benchmark.json

# Run only compute benchmarks
ornn-bench run --compute-only

# Run multiple selective sections
ornn-bench run --compute-only --memory-only

# Run and upload results
ornn-bench run --upload --api-key YOUR_KEY

# Using environment variable for API key
export ORNN_API_KEY=YOUR_KEY
ornn-bench run --upload
```

**Execution Order:**

Benchmarks execute in a deterministic order following the Section 8 runbook:

1. **Pre-flight** — GPU UUID collection, NVLink topology, driver/CUDA versions, system inventory
2. **Compute** — MAMF benchmarks (BF16, FP8 E4M3, FP8 E5M2, FP16, TF32, fixed-shape matmul)
3. **Memory** — nvbandwidth tests (device local CE/SM, H2D, D2H, D2D read/write, D2D bidirectional) + PyTorch D2D cross-validation
4. **Interconnect** — NCCL tests (all-reduce sweep, 1GB variance, reduce-scatter, all-gather, broadcast, send-receive)
5. **Monitoring** — Thermal/power monitoring snapshots, XID error detection
6. **Post-flight** — UUID consistency, NVLink error check, ECC error check
7. **Manifest** — Output artifact inventory

Infrastructure sections (pre-flight, monitoring, post-flight, manifest) always run regardless of scope filters.

**Exit Codes:**

| Code | Meaning |
|------|---------|
| `0` | All sections completed successfully |
| `1` | Fatal error (no GPU, missing API key, upload failure) |
| `2` | Partial failure (some sections failed but report was generated) |

**Partial Failure Behavior:**

If an individual benchmark fails or times out, the CLI:
- Records the failure status in the report
- Continues with remaining eligible sections
- Displays a partial failure summary
- Exits with code `2`
- Never shows raw Python tracebacks to the user

---

### `ornn-bench info`

Display system and GPU environment diagnostics.

```bash
ornn-bench info
```

This command works on any machine, including those without NVIDIA GPUs or benchmark tools.

**Output includes:**

- **System** — OS, kernel, CPU model
- **GPU** — GPU count, model names, driver/CUDA versions (or remediation if missing)
- **Python** — Python version, PyTorch availability and CUDA support
- **Benchmark Tools** — Availability status of nvidia-smi, nvbandwidth, nccl-tests with installation guidance for missing tools

**Example output (no GPU):**

```
╭─── System ───╮
│  OS      Darwin 23.5.0  │
│  Kernel  23.5.0         │
│  CPU     Apple M2 Pro   │
╰──────────────╯

╭─── GPU ───╮
│  nvidia-smi not found on PATH                       │
│  Install NVIDIA GPU drivers. Verify with: nvidia-smi │
╰───────────╯

╭─── Python ───╮
│  Python   3.12.3             │
│  PyTorch  not installed      │
╰──────────────╯

╭─── Benchmark Tools ───╮
│  nvidia-smi          ✗ missing   Install NVIDIA GPU drivers     │
│  nvbandwidth         ✗ missing   Build from NVIDIA/nvbandwidth  │
│  nccl-tests          ✗ missing   Build from NVIDIA/nccl-tests   │
╰────────────────────────╯
```

---

### `ornn-bench report`

Re-render a previously saved benchmark report. Works on machines without GPU tools.

```bash
ornn-bench report REPORT_FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `REPORT_FILE` | Path to a JSON report file |

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as machine-readable JSON (no ANSI codes, no progress bars) |
| `--plain` | Output as plain text (no ANSI codes, no box-drawing characters) |

**Examples:**

```bash
# Rich terminal display (default)
ornn-bench report ornn_report_abc12345.json

# Machine-readable JSON output
ornn-bench report ornn_report_abc12345.json --json

# Plain text (piping-safe)
ornn-bench report ornn_report_abc12345.json --plain

# Pipe to a file
ornn-bench report ornn_report_abc12345.json --json > parsed.json
```

**Output Modes:**

| Mode | When to Use |
|------|-------------|
| Default (Rich) | Interactive terminal sessions |
| `--json` | Piping to other tools, CI/CD, automated processing |
| `--plain` | Non-TTY environments, log files, text editors |

Both `--json` and `--plain` modes produce output free of ANSI escape codes and progress bar artifacts, making them safe for piping and machine processing.

---

### `ornn-bench upload`

Upload a saved benchmark report to the Ornn API.

```bash
ornn-bench upload REPORT_FILE [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `REPORT_FILE` | Path to a JSON report file to upload |

**Options:**

| Option | Description |
|--------|-------------|
| `--api-key TEXT` | API key for authentication (also via `ORNN_API_KEY` env var) |
| `--verify / --no-verify` | Verify local vs server scores after upload (default: `--verify`) |

**Examples:**

```bash
# Upload with API key
ornn-bench upload ornn_report_abc12345.json --api-key YOUR_KEY

# Using environment variable
export ORNN_API_KEY=YOUR_KEY
ornn-bench upload ornn_report_abc12345.json

# Upload without score verification
ornn-bench upload ornn_report_abc12345.json --api-key YOUR_KEY --no-verify
```

**Pre-upload Validation:**

Before sending the report to the API, the CLI validates:
- The report file exists and is valid JSON
- The report has a valid schema version
- Required fields are present

If validation fails, a clear error message is shown and no network call is made.

**Error Messages:**

| Error | Cause | Guidance |
|-------|-------|----------|
| Missing API Key | No `--api-key` flag or `ORNN_API_KEY` env var | Set the API key |
| Authentication Error | Invalid or revoked API key | Check your key |
| Schema Version Error | Report uses unsupported schema version | Upgrade/downgrade ornn-bench |
| Validation Error | Report payload rejected by the API | Check report format |
| Rate Limit Exceeded | Too many requests in the time window | Wait and retry |
| Network Error | Connection failure | Check network; retries are safe |

**Retry Safety:**

Retrying after a failure is always safe. The API uses idempotent ingest to prevent duplicate uploads. The same report can be submitted multiple times without creating duplicate entries.

---

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `ORNN_API_URL` | API base URL | `run --upload`, `upload` |
| `ORNN_API_KEY` | API key for authentication | `run --upload`, `upload` |

---

## Output Formats

### JSON Report

Every `ornn-bench run` produces a JSON report file containing:

```json
{
  "schema_version": "1.0.0",
  "report_id": "unique-uuid",
  "created_at": "2025-01-15T10:30:00Z",
  "system_inventory": { ... },
  "sections": [
    {
      "name": "compute",
      "status": "completed",
      "started_at": "...",
      "finished_at": "...",
      "metrics": { ... }
    }
  ],
  "scores": {
    "ornn_i": 95.2,
    "ornn_t": 88.7,
    "qualification": "Premium",
    "components": { ... },
    "score_status": "valid"
  },
  "manifest": { ... }
}
```

### Terminal Scorecard

The default terminal output displays:
- **Ornn Scores** — Color-coded Ornn-I and Ornn-T values
- **Component Metrics** — Individual BW, FP8, BF16, AR values
- **Per-GPU Scores** — Individual GPU scores for multi-GPU systems
- **Qualification** — Grade badge (Premium/Standard/Below) with status

---

## Troubleshooting

### No NVIDIA GPU detected

```
ornn-bench run
# ╭── No NVIDIA GPU Detected ──╮
# │ nvidia-smi not found ...    │
# ╰─────────────────────────────╯
```

**Fix:** Ensure you are on a machine with an NVIDIA GPU and drivers installed. Run `nvidia-smi` to verify.

### Benchmark tool not found

```
ornn-bench info
# nvbandwidth    ✗ missing    Build from NVIDIA/nvbandwidth
```

**Fix:** Install the missing tool:
- **nvbandwidth**: Build from [NVIDIA/nvbandwidth](https://github.com/NVIDIA/nvbandwidth) (requires CUDA toolkit + CMake)
- **nccl-tests**: Build from [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) (requires NCCL library + MPI)
- **mamf-finder.py**: Get from [stas00/ml-engineering](https://github.com/stas00/ml-engineering)

### Upload authentication failure

```
╭── Authentication Error ──╮
│ Unauthorized.             │
│ Check your API key ...    │
╰───────────────────────────╯
```

**Fix:** Verify your API key is correct and not revoked. Set it via `--api-key` or `ORNN_API_KEY`.

### Rate limit exceeded

```
╭── Rate Limit Exceeded ──╮
│ Too many requests.       │
│ Retry after 15 seconds.  │
╰──────────────────────────╯
```

**Fix:** Wait the indicated number of seconds before retrying. The default limit is 60 requests per 60-second window.

### Partial benchmark failure

If some benchmarks fail, the CLI continues with remaining sections and exits with code `2`. Review the JSON report for per-section status details.

### Piped output has ANSI codes

Use `--json` or `--plain` flags:

```bash
ornn-bench report results.json --json | jq .
ornn-bench report results.json --plain > output.txt
```
