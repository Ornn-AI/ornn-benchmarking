# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** env vars, external credentials/services, host prerequisites.
**What does NOT belong here:** service ports/commands (use `.factory/services.yaml`).

---

## Required Runtime Dependencies

- Python 3.10+
- NVIDIA GPU drivers + `nvidia-smi` (for real benchmark execution)
- CUDA toolkit and PyTorch compatible with target GPU
- `nvbandwidth` binary
- `nccl-tests` binaries

## API/Cloud Dependencies

- GCP project: `ornn-benchmarking`
- Cloud Run API deployment target
- Firestore (native mode, `us-east1`)

## Environment Variables (Planned)

- `ORNN_API_BASE_URL` — Upload API base URL
- `ORNN_API_KEY` — API key for upload/verify endpoints
- `FIRESTORE_PROJECT_ID` — GCP project id used by API
- `FIRESTORE_EMULATOR_HOST` — Local emulator host for development/testing

## Notes

- Mission must not modify existing Supabase projects.
- Keep infrastructure free-tier-safe by default.
