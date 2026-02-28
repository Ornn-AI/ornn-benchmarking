# mamf-finder.py Research

**Source**: https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py
**Full Name**: Maximum Achievable Matmul FLOPS (MAMF) Finder
**Current benchmark_version**: 2

## Installation / How to Run

Single-file Python script. No package installation needed, just download and run:

```bash
wget https://raw.githubusercontent.com/stas00/ml-engineering/refs/heads/master/compute/accelerator/benchmarks/mamf-finder.py
python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096 --output_file=results.txt
```

### Dependencies
- `torch` (PyTorch)
- `numpy`
- `packaging`

### Supported Accelerators
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- HPU (Intel Gaudi)
- XPU (Intel dGPUs like ARC A770)
- MPS (Apple Silicon)

## CLI Arguments

### Matrix Dimension Arguments (mutually exclusive groups, all three required)

**M dimension** (first dimension of GEMM):
- `--m <val1> [val2] ...` — Explicit M values (nargs="+")
- `--m_range <start> <stop> <step>` — Range specification [start, stop, step]

**N dimension** (last dimension of GEMM):
- `--n <val1> [val2] ...` — Explicit N values (nargs="*")
- `--n_range <start> <stop> <step>` — Range specification [start, stop, step]

**K dimension** (shared/reduction dimension):
- `--k <val1> [val2] ...` — Explicit K values (nargs="*")
- `--k_range <start> <stop> <step>` — Range specification [start, stop, step]

**Note**: If range start is 0, it's automatically bumped to `step` (can't have a 0 dimension).

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_iterations` | int | 100 | Number of iterations used to benchmark each GEMM |
| `--num_warmup_iterations` | int | 50 | Number of warmup iterations |
| `--cuda_device` | int | 0 | CUDA device to run benchmark on |
| `--output_file` | str | `{script_dir}/results/mm.out` | Output file path |
| `--notes` | str | "" | Benchmark-specific notes for header |
| `--verbose` / `--no-verbose` | bool | True | Log to stdout besides output_file |
| `--dtype` | str | "bfloat16" | Data type for benchmark |

## Supported dtypes

The `--dtype` argument accepts any valid `torch` dtype name string:

| dtype string | torch dtype | Notes |
|-------------|-------------|-------|
| `bfloat16` | `torch.bfloat16` | Default |
| `float16` | `torch.float16` | |
| `float32` | `torch.float32` | |
| `float8_e4m3fn` | `torch.float8_e4m3fn` | NVIDIA only, requires torch>=2.5. Uses `torch._scaled_mm` |
| `float8_e4m3fnuz` | `torch.float8_e4m3fnuz` | AMD ROCm only, requires torch>=2.5. Uses `torch._scaled_mm` |

**Important**: There is no "tf32" dtype string — TF32 is automatically used by PyTorch when computing in float32 on Ampere+ GPUs (controlled by `torch.backends.cuda.matmul.allow_tf32`).

## Output Format

### Header
The output file starts with a metadata header:
```
Benchmark started on 2024-01-15 10:30:00

** Command line:
python mamf-finder.py --m_range 0 20480 256 --n 4096 --k 4096

** Dtype: torch.bfloat16

** Platform/Device info:
- Linux hostname 5.15.0 ...
- _CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', ...)

** Critical software versions:
- torch=2.1.0
- cuda=12.1

** Critical environment variables:
- PYTORCH_TUNABLEOP_ENABLED=0

** Additional notes:
- benchmark version: 2

--------------------------------------------------------------------------------
```

### Per-shape output line (printed with `\r` for live update)
```
     1 |  756.1(mean)  758.2(median)  762.3(max) @ 256x4096x4096        | best:  756.1(mean)  758.2(median)  762.3(max) TFLOPS
```

Format: `{num_shapes:>6} | {mean_tflops:6.1f}(mean) {median_tflops:6.1f}(median) {max_tflops:6.1f}(max) @ {MxNxK:<20} | best: {best_mean:6.1f}(mean) {best_median:6.1f}(median) {best_max:6.1f}(max) TFLOPS`

### Summary (at end)
```
Tried 80 shapes => the best outcomes were:
mean:   756.1 TFLOPS @ 8192x4096x4096 (MxNxK)
median: 758.2 TFLOPS @ 8192x4096x4096 (MxNxK)
max:    762.3 TFLOPS @ 8192x4096x4096 (MxNxK)

Across 80 shapes in range: m=[0, 20480, 256] | n=[4096] | k=[4096] in this run:
arithmetic mean: 650.2 TFLOPS
geometric mean:  640.1 TFLOPS

Legend: TFLOPS = 10**12 FLOPS
Elapsed time: 1:23:45
```

## Key Metrics Reported
- **mean TFLOPS**: Mean of all iteration times → TFLOPS
- **median TFLOPS**: Median of all iteration times → TFLOPS
- **max TFLOPS**: Minimum time (=maximum TFLOPS)
- **arithmetic mean** (across all shapes)
- **geometric mean** (across all shapes)
- Best config (MxNxK shape) for each metric

## Integration Considerations

1. **Output parsing**: The output uses `\r` carriage returns for live progress. The `Tee` class strips `\r` and ANSI escape codes from file output, replacing with `\n`.

2. **Signal handling**: Supports SIGINT (Ctrl+C) graceful shutdown — reports best results found so far.

3. **Accelerator warmup**: 30-second warmup period before benchmarking begins.

4. **Cache clearing**: Between each iteration, writes zeros to a 256MB buffer to flush L2 cache, and re-randomizes the destination matrix C.

5. **FP8 handling**: Uses `torch._scaled_mm` instead of `torch.mm` for FP8 dtypes. Scale is always `torch.tensor([1.0])`.

6. **FLOPS calculation**: `flos = 2 * m * n * k` (standard matmul FLOP count).

7. **Output file**: Uses the `Tee` class to write to both file and stdout simultaneously.

8. **No structured output format**: The output is human-readable text only — no JSON or CSV option. Must be parsed from text.
