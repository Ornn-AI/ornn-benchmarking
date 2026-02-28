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

## Setup Notes

- Default local API port: `8080`
- Firestore emulator port: `8085`
- On non-GPU hosts, validate behavior using fixture-driven runs and dependency-missing paths.

## Known Limitations

- Real GPU benchmark execution cannot be fully validated on macOS host without NVIDIA GPU toolchain.
- Full end-to-end hardware validation must be run on Linux NVIDIA instances.
