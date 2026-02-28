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
- Python venv: `./.venv` at repo root `/Users/kushbavaria/ornn-benchmarking`
- Install: `./.venv/bin/python -m pip install -e ".[dev]"`
- Test fixtures are in `tests/fixtures/sample_report.json` and `tests/fixtures/sample_scored_report.json`

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
