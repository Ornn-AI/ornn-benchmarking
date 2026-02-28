# Contributing

Thank you for your interest in contributing to the Ornn GPU Benchmarking CLI! This guide covers development setup, code style, testing, and the pull request workflow.

## Development Setup

### Prerequisites

- **Python 3.10+**
- **Git**

> **Note:** GPU hardware is **not required** for development. All GPU interactions are mocked in tests.

### Clone and Install

```bash
git clone https://github.com/Ornn-AI/ornn-benchmarking.git
cd ornn-benchmarking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install in development mode with all dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# CLI is available
ornn-bench --version

# Tests pass
pytest -q

# Linting passes
ruff check .

# Type checking passes
mypy src api
```

---

## Project Structure

```
src/ornn_bench/     # CLI package (installed as ornn-bench)
api/                # FastAPI backend (deployed to Cloud Run)
tests/              # Test suite
docs/               # Documentation
```

See [Architecture](docs/architecture.md) for detailed component descriptions.

---

## Code Style

### Formatter and Linter

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting.

```bash
# Check for lint issues
ruff check .

# Auto-fix issues
ruff check . --fix
```

**Configuration** is in `pyproject.toml` under `[tool.ruff]`:

- Line length: 100 characters
- Target: Python 3.10
- Rules: pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, flake8-simplify, ruff-specific

### Type Checking

This project uses [mypy](https://mypy.readthedocs.io/) for static type checking.

```bash
mypy src api
```

All functions must have type annotations (`disallow_untyped_defs = true`).

### Conventions

- **Docstrings:** Use Google-style docstrings for modules, classes, and public functions
- **Imports:** Sorted by isort via Ruff. Use `from __future__ import annotations` at the top of each module
- **Constants:** Module-level `UPPER_SNAKE_CASE`
- **Private functions:** Prefix with underscore (`_helper_function`)
- **Type annotations:** Use modern syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`)
- **No secrets:** Never commit API keys, tokens, or credentials — not even in test fixtures

---

## Testing

### Running Tests

```bash
# All tests
pytest -q

# Verbose output
pytest -v

# Specific test file
pytest tests/test_scoring.py

# Specific test by name
pytest -k "test_compute_ornn_i"

# With coverage (if pytest-cov is installed)
pytest --cov=ornn_bench --cov=api
```

### Test Organization

```
tests/
├── api/           # API endpoint tests (FastAPI TestClient)
├── cli/           # CLI command tests (Typer CliRunner)
├── runbook/       # Benchmark runner/parser tests
├── scoring/       # Scoring engine tests
├── fixtures/      # Sample tool outputs for deterministic testing
├── integration/   # End-to-end flow tests
├── release/       # Packaging and release tests
├── conftest.py    # Shared fixtures and mock Firestore client
├── test_models.py # Data model tests
├── test_scoring.py # Core scoring tests
└── test_smoke.py  # Import smoke tests
```

### Writing Tests

- **Mock external tools:** All GPU tools (`nvidia-smi`, `nvbandwidth`, `nccl-tests`) must be mocked. Use `monkeypatch` or `unittest.mock.patch` on subprocess calls
- **Use fixtures:** Store sample tool outputs in `tests/fixtures/` for deterministic parsing tests
- **Mock Firestore:** Tests use an in-memory mock Firestore client defined in `tests/conftest.py`. No real Firestore or Java emulator needed
- **Test error paths:** Include tests for failure scenarios (missing tools, invalid inputs, network errors, auth failures)

### Test Fixtures

Sample benchmark outputs are stored in `tests/fixtures/`:

```bash
tests/fixtures/
├── nvbandwidth/          # Sample nvbandwidth JSON outputs
├── nccl/                 # Sample NCCL test outputs
└── mamf/                 # Sample MAMF finder outputs
```

Use these for deterministic parsing and scoring tests without requiring GPU hardware.

---

## Pull Request Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature main
```

Use descriptive branch names:
- `feature/add-new-benchmark-type`
- `fix/scoring-nan-handling`
- `docs/update-api-reference`

### 2. Make Changes

- Keep changes focused — one feature or fix per PR
- Follow existing code patterns and style
- Add tests for new functionality
- Update documentation if behavior changes

### 3. Verify Before Submitting

```bash
# Run the full validation suite
ruff check .          # Lint
mypy src api          # Type check
pytest -q             # Tests
```

All three must pass before submitting.

### 4. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat(cli): add --timeout flag for benchmark sections
fix(scoring): handle NaN inputs in Ornn-I computation
docs(api): document rate limit headers
test(runbook): add fixture for nvbandwidth edge case
chore: update ruff to 0.5.0
```

Prefixes: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`

### 5. Open a Pull Request

- Target the `main` branch
- Describe what changed and why
- Reference related issues if applicable
- Ensure CI checks pass

---

## Issue Reporting

When filing an issue, include:

- **OS and Python version** (`python --version`)
- **ornn-bench version** (`ornn-bench --version`)
- **GPU info** (if relevant — `ornn-bench info` output)
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Full error output** (use `--plain` mode if terminal rendering is garbled)

---

## Architecture Decisions

Before proposing significant changes to the architecture, please:

1. Read the [Architecture document](docs/architecture.md) to understand current design
2. Open an issue to discuss the proposed change
3. Consider the free-tier-safe constraint (no paid always-on infrastructure)
4. Ensure backward compatibility with the `1.0.0` schema version

---

## Local API Development

To work on the API backend locally:

```bash
# Start the API (mock Firestore, no external deps needed)
PORT=8080 python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/api/v1/runs \
  -H "X-API-Key: dev-test-key" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

See the [Deployment Guide](docs/deployment.md) for Firestore emulator setup and Docker options.
