# Benchmark Campaign

Campaign-specific notes for running OrnnBench across providers.

**What belongs here:** toolchain lock conventions, pin resolution notes, report naming conventions, and operational gotchas found during runs.
**What does NOT belong here:** service start/stop commands (use `.factory/services.yaml`).

---

## 2026-03-05 — bench-infra folder bootstrap

- `ornn-bench-infra/toolchain.env` now exists with the accepted pins from `mission.md`:
  - `ORNN_BENCH_VERSION=0.1.0`
  - `MAMF_FINDER_COMMIT=7c660da71e533fdb5de141591379d3c8070ef272`
  - `NVBANDWIDTH_TAG=v0.8` (tag comment records commit `66746a3bef61c8c2e12ab34955310da70b9e38cb`)
  - `NCCL_TESTS_COMMIT=ae98985f5599617be94042f4aa3637d10014ce89`
- `ornn-bench-infra/provision.sh` and `ornn-bench-infra/benchmark_node.sh` are currently strict-mode executable scaffolds that intentionally exit non-zero until the follow-up bench-infra features implement them.
- Primary provider placeholder notes now exist under `ornn-bench-infra/provider-notes/{aws,gcp,azure,oracle}.md`.

## 2026-03-06 — provision.sh runtime contract

- `ornn-bench-infra/provision.sh` now assumes a **Linux + apt + root** provider image and hard-fails early otherwise.
- NCCL is treated as the strict lock gate: the script checks `apt-cache madison` for `libnccl2=${NCCL_DEV_VERSION}` and `libnccl-dev=${NCCL_DEV_VERSION}` before install, then exits non-zero with an actionable CUDA-repo message if either pin is unavailable.
- Provisioned artifacts live under `/opt/ornn-bench-tools/` with source checkouts in `src/`, binaries/symlinks in `bin/`, and the Python venv in `venv/`.
- `/opt/ornn-bench-tools/runtime.env` is the handoff file for the node runner and currently exports: `ORNN_BENCH_BIN`, `MAMF_FINDER_PATH`, `NVBANDWIDTH_PATH`, `ALL_REDUCE_PERF_PATH`, `REDUCE_SCATTER_PERF_PATH`, `ALL_GATHER_PERF_PATH`, `BROADCAST_PERF_PATH`, `SENDRECV_PERF_PATH`, plus `CUDA_*` and `NCCL_*` directories.
