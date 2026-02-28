# Benchmark Tooling

Reference notes for benchmark tool integration.

---

## mamf-finder.py

- Source: `stas00/ml-engineering`
- Expected usage: range scans (`--m_range --n_range --k_range`) and fixed-shape runs (`--m --n --k`)
- Output: text format (requires parser)
- Dtypes to support in mission: BF16, FP8 E4M3, FP8 E5M2, FP16, optional TF32 path

## nvbandwidth

- Source: `NVIDIA/nvbandwidth`
- Output mode: JSON (`-j/--json`) with matrix values
- Required test matrix: device local CE/SM, H2D, D2H, D2D read/write, D2D bidirectional

## nccl-tests

- Source: `NVIDIA/nccl-tests`
- Required binaries: all_reduce_perf, reduce_scatter_perf, all_gather_perf, broadcast_perf, sendrecv_perf
- Primary metric: bus bandwidth (`busbw`) including fixed 1GB variance run

## Parser/Runner Expectations

- Preserve raw output references in report for auditability
- Emit deterministic parsed structures for scoring/validation tests
- Clearly mark unsupported/missing-tool scenarios as skipped/failed with reason
