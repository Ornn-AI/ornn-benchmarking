# nccl-tests Research

**Source**: https://github.com/NVIDIA/nccl-tests
**License**: BSD-3-Clause
**Latest commit**: Feb 9, 2026

## Installation / Build

### Requirements
- CUDA toolkit
- NCCL library
- (Optional) MPI for multi-node

### Basic Build
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j
```

### With custom paths
```bash
make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

### With MPI support
```bash
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

### With name suffix
```bash
make MPI=1 NAME_SUFFIX=_mpi
```

### Output binaries
All built to `./build/` directory:
- `all_reduce_perf`
- `all_gather_perf`
- `reduce_scatter_perf`
- `broadcast_perf`
- `reduce_perf`
- `sendrecv_perf`
- `alltoall_perf`
- `scatter_perf`
- `gather_perf`
- `hypercube_perf`

## CLI Arguments (all test binaries share same arguments)

### GPU Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `-t, --nthreads <N>` | 1 | Number of threads per process |
| `-g, --ngpus <N>` | 1 | Number of GPUs per thread |

Total ranks = (num processes) × (nthreads) × (ngpus)

### Size Scanning
| Argument | Default | Description |
|----------|---------|-------------|
| `-b, --minbytes <size>` | 32M | Minimum size to start with |
| `-e, --maxbytes <size>` | 32M | Maximum size to end at |
| `-i, --stepbytes <size>` | 1M | Fixed increment between sizes |
| `-f, --stepfactor <N>` | disabled | Multiplication factor between sizes |

**Size suffixes**: K (KiB=1024), M (MiB=1024²), G (GiB=1024³)

**Note**: `-i` and `-f` are mutually exclusive. Use one or the other.

### NCCL Operation Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `-o, --op <op/all>` | sum | Reduction operation (sum/prod/min/max/avg/all) |
| `-d, --datatype <type/all>` | float | Data type (nccltype or "all") |
| `-r, --root <root/all>` | 0 | Root rank (for broadcast/reduce) |

### Performance Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `-n, --iters <N>` | 20 | Number of timed iterations |
| `-w, --warmup_iters <N>` | 1 | Number of warmup iterations (not timed) |
| `-m, --agg_iters <N>` | 1 | Operations to aggregate per iteration |
| `-N, --run_cycles <N>` | 1 | Run & print each cycle (0=infinite) |
| `-a, --average <0/1/2/3>` | 1 | MPI averaging: 0=Rank0, 1=Avg, 2=Min, 3=Max |

### Test Operation Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `-p, --parallel_init <0/1>` | 0 | Use threads for parallel NCCL init |
| `-c, --check <N>` | 1 | Check correctness every N iterations |
| `-z, --blocking <0/1>` | 0 | Make NCCL collective blocking |
| `-G, --cudagraph <N>` | 0 | Capture as CUDA graph, replay N times |
| `-C, --report_cputime <0/1>` | 0 | Report CPU time instead of latency |
| `-R, --local_register <0/1/2>` | 0 | Buffer registration: 0=off, 1=local, 2=symmetric |
| `-S, --report_timestamps <0/1>` | 0 | Add timestamp to each report line |
| `-J, --output_file <file>` | none | Write JSON output to filepath (infer from .json suffix) |
| `-T, --timeout <seconds>` | disabled | Timeout per test |
| `-M, --memory_report <0/1>` | 0 | Enable memory usage report |

## Usage Examples

### Single node, 8 GPUs, scanning sizes
```bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
```

### Multi-node (64 GPUs across 8 nodes)
```bash
mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

### With specific iterations
```bash
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8 -n 100 -w 5
```

## Output Format (stdout)

### Header
```
# nThread 1 nGpus 8 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  12345 on   hostname device  0 [0x07] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid  12345 on   hostname device  1 [0x0a] NVIDIA H100 80GB HBM3
...
```

### Data Table
```
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    33.52    0.00    0.00      0    32.12    0.00    0.00      0
          16             4     float     sum      -1    32.80    0.00    0.00      0    32.45    0.00    0.00      0
          32             8     float     sum      -1    33.11    0.00    0.00      0    33.55    0.00    0.00      0
...
   134217728      33554432     float     sum      -1   1234.5  108.72  190.26      0   1230.2  109.10  190.93      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 85.1234
```

### Column Descriptions
| Column | Description |
|--------|-------------|
| `size` | Data size in bytes |
| `count` | Number of elements |
| `type` | Data type (float, half, double, etc.) |
| `redop` | Reduction operation |
| `root` | Root rank (-1 for all-reduce) |
| `time` | Average operation time in microseconds (μs) |
| `algbw` | Algorithm bandwidth in GB/s = size / time |
| `busbw` | Bus bandwidth in GB/s (hardware-corrected) |
| `#wrong` | Number of incorrect values (correctness check) |

### Parsing the busbw Column

The **busbw** (bus bandwidth) is the key metric. It's the algorithm bandwidth multiplied by a correction factor that accounts for the number of ranks:

| Operation | Bus BW Formula | Factor |
|-----------|---------------|--------|
| AllReduce | `algbw * 2*(n-1)/n` | 2*(n-1)/n |
| ReduceScatter | `algbw * (n-1)/n` | (n-1)/n |
| AllGather | `algbw * (n-1)/n` | (n-1)/n |
| Broadcast | `algbw * 1` | 1 |
| Reduce | `algbw * 1` | 1 |
| AlltoAll | `algbw * (n-1)/n` | (n-1)/n |

Where `n` = number of ranks.

The busbw should reflect the actual hardware link speed (NVLink, PCIe, network) independent of rank count.

### Summary Line
```
# Avg bus bandwidth    : 85.1234
```

This is the average busbw across all tested sizes.

## JSON Output Format (-J flag)

When using `-J output.json`, nccl-tests writes a JSON file. The format includes:
- Test configuration
- Per-size results with all columns
- Summary statistics

## Test Binaries and Their Operations

| Binary | NCCL Operation | Description |
|--------|---------------|-------------|
| `all_reduce_perf` | AllReduce | Most common: reduce + broadcast across all ranks |
| `reduce_scatter_perf` | ReduceScatter | Reduce, scatter result chunks to ranks |
| `all_gather_perf` | AllGather | Gather chunks from all ranks to all |
| `broadcast_perf` | Broadcast | Copy from root to all ranks |
| `reduce_perf` | Reduce | Reduce to root rank |
| `sendrecv_perf` | Send/Recv | Point-to-point between pairs |
| `alltoall_perf` | AllToAll | Each rank sends different data to each other rank |
| `scatter_perf` | Scatter | Scatter from root to all |
| `gather_perf` | Gather | Gather to root from all |
| `hypercube_perf` | Hypercube | Hypercube collective pattern |

## Environment Variables for Partitioning

- `NCCL_TESTS_SPLIT` — Partition GPUs into groups (syntax: `<op><value>`, op=AND|OR|MOD|DIV or &|%|/)
- `NCCL_TESTS_SPLIT_MASK` — Equivalent to `NCCL_TESTS_SPLIT="&<value>"`

Examples:
- `NCCL_TESTS_SPLIT="MOD 8"` — 8 parallel single-GPU operations (one per node)
- `NCCL_TESTS_SPLIT="DIV 8"` — One operation per node (intra-node only)

## Integration Considerations

1. **Output parsing**: Lines starting with `#` are comments/headers. Data lines have no prefix. The table is space-separated with fixed-width columns.

2. **Key metric**: The `busbw` column (column index 8 or 12 for out-of-place/in-place) is the primary metric for benchmarking.

3. **The last line** `# Avg bus bandwidth : X.XXXX` gives the summary metric.

4. **Size suffixes**: Input `-b` and `-e` accept K/M/G suffixes (binary: KiB, MiB, GiB).

5. **Multiple ops**: Use `-o all` to test all reduction operations. Use `-d all` to test all datatypes.

6. **Typical benchmark command**:
   ```bash
   ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 20 -w 5
   ```

7. **MPI requirement**: Multi-node requires MPI. Single-node uses `-g` for GPU count. Total ranks = processes × threads × GPUs.

8. **No JSON by default**: Must use `-J <file>.json` to get JSON output. Otherwise, parsing stdout text is required.

9. **Exit code**: Returns 0 on success.

10. **stderr vs stdout**: All benchmark output goes to stdout. Errors go to stderr.

11. **Correctness check**: The `#wrong` column should always be 0. Non-zero indicates data corruption.

12. **Important for parsing**: Both out-of-place AND in-place results are shown. Each has its own time/algbw/busbw/#wrong columns. Usually, people report out-of-place busbw.

13. **The data line format** (regex-parseable):
    ```
    ^\s+(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$
    ```
    Groups: size, count, type, redop, root, oop_time, oop_algbw, oop_busbw, oop_wrong, ip_time, ip_algbw, ip_busbw, ip_wrong
