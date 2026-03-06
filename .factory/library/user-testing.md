# User Testing Surface

Manual validation notes for user-facing behavior.

---

## Surfaces

1. **CLI surface** (terminal)
   - `ornn-bench --help`
   - `ornn-bench info`
   - `ornn-bench run` (fixture/mock mode on non-GPU host)
   - `ornn-bench report <json>`
   - `ornn-bench upload <json>`

2. **API surface** (HTTP)
   - `POST /api/v1/runs`
   - `GET /api/v1/runs/{id}`
   - `POST /api/v1/verify`
   - `GET /health`

## Tools

- Terminal + curl for manual API checks
- pytest for automated assertions
- Fixtures/mocks for GPU-tool output on non-GPU development hosts
- `tuistory` skill for terminal UI automation and snapshot capture

## Setup Notes

- Default local API port: `8080`
- Firestore emulator port: `8085`
- On non-GPU hosts, validate behavior using fixture-driven runs and dependency-missing paths.
- CLI entrypoint: `./.venv/bin/ornn-bench`
- Python venv: `./.venv` at repo root `/Users/kushbavaria/Documents/ornn-benchmarking`
- Install: `./.venv/bin/python -m pip install -e ".[dev]"`
- Test fixtures are in `tests/fixtures/sample_report.json` and `tests/fixtures/sample_scored_report.json`
- The sample report fixtures keep top-level `manifest` empty; validate manifest mapping with `tests/runbook/test_manifest_and_durability.py` or a temporary enriched report fixture instead of relying on `sample_report.json` alone.
- Bench-core validation on this macOS host does **not** require starting local services; rely on fixture-driven CLI commands and pytest coverage for GPU-specific paths.

## Flow Validator Guidance: API

### Testing tool
Use `curl` (directly via Execute tool) for all API endpoint testing. The test API server runs on `http://localhost:8080` with an in-memory fake Firestore backend.

### Credentials / API keys
- **Valid keys:** `test-key-1`, `test-key-2`, `test-key-3`
- **Revoked key:** `revoked-key-1`
- Header: `X-API-Key: <key>`

### Rate limiting
The test server is configured with a rate limit of **3 requests per 10 seconds per key**. Each subagent MUST use its own dedicated key to avoid cross-contamination of rate-limit state. If you need more than 3 requests with one key, wait 10 seconds between batches of 3.

### Isolation rules
- Each subagent uses a **different API key** for isolation
- Each subagent uses **unique `report_id` values** (prefix with subagent group name)
- The test server uses in-memory storage — data does not persist across restarts
- Subagents MUST NOT restart the API server
- Subagents MUST NOT modify API source code

### Endpoints
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | /health | No | Health check |
| POST | /api/v1/runs | Yes | Submit benchmark run |
| GET | /api/v1/runs/{id} | Yes | Retrieve run by ID |
| POST | /api/v1/verify | Yes | Verify scores against server recomputation |

### Valid payload template
```json
{
  "schema_version": "1.0.0",
  "report_id": "<unique-id>",
  "created_at": "2024-01-15T10:30:00Z",
  "system_inventory": {
    "gpus": [{"uuid": "GPU-test", "name": "Test GPU", "driver_version": "535.0", "cuda_version": "12.2", "memory_total_mb": 81920}],
    "os_info": "Ubuntu 22.04", "kernel_version": "5.15.0", "cpu_model": "Xeon", "numa_nodes": 1, "pytorch_version": "2.1.0"
  },
  "sections": [],
  "scores": {
    "ornn_i": 100.0, "ornn_t": 100.0, "qualification": "Premium",
    "components": {"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
    "score_status": "valid"
  },
  "manifest": {}
}
```

### Supported schema version
Only `1.0.0` is supported. Any other version gets 422.

## Flow Validator Guidance: CLI Upload

### Testing tool
Use `./.venv/bin/ornn-bench upload <file>` via Execute tool.

### Setup
- The API server must be running on `http://localhost:8080`
- Set `ORNN_API_URL=http://localhost:8080` and `ORNN_API_KEY=<key>` when running CLI upload commands
- Sample report files: `tests/fixtures/sample_report.json`, `tests/fixtures/sample_scored_report.json`

### Isolation rules
- Each subagent uses its own API key
- Unique report_id values per subagent
- Do not modify fixtures

## Known Limitations

- Real GPU benchmark execution cannot be fully validated on macOS host without NVIDIA GPU toolchain.
- Full end-to-end hardware validation must be run on Linux NVIDIA instances.
- `upload` command is not yet implemented (returns exit code 1 with message). This is expected for core-cli milestone.

## Flow Validator Guidance: CLI

### Testing tool
Use the `tuistory` skill for TUI/terminal interaction, or direct shell execution via Execute tool for simpler command-output assertions.

### Isolation rules
- All subagents use the same CLI binary at `./.venv/bin/ornn-bench`
- CLI testing is read-only (no shared mutable state between subagents)
- Report file fixtures in `tests/fixtures/` are read-only shared resources
- Each subagent should write any temporary output files to unique names to avoid collision
- When running pytest in parallel validator sessions, use a unique `--basetemp=/tmp/<namespace>` per subagent
- Subagents MUST NOT modify source code or test fixtures

### What's testable on non-GPU macOS host
- `--version`, `--help`, `run --help` (all commands)
- `info` command (shows missing GPU remediation)
- `run` command (exits with no-GPU guardrail message, exit code 1)
- `run --compute-only` (same no-GPU guardrail)
- `report <file>` (re-renders saved reports, works without GPU)
- `report <file> --json` (JSON output mode)
- `report <file> --plain` (plain text mode)
- Piped output (pipe through `cat` to test non-TTY behavior)
- Automated tests via `pytest` validate runbook structure, scoring, parsing

### What's NOT testable on non-GPU macOS host
- Actual live benchmark execution (requires nvidia-smi, CUDA)
- Live progress bars during benchmark run
- Real GPU metrics and actual scoring from live data
- These are validated through automated unit tests with mocked subprocess calls
