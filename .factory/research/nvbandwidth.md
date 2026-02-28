# nvbandwidth Research

**Source**: https://github.com/NVIDIA/nvbandwidth
**Version**: v0.8 (April 2025)
**License**: Apache-2.0

## Installation / Build

### Requirements
- CUDA toolkit (version 11.X or above; multinode requires 12.3+ and driver 550+)
- C++17 compiler (GCC 7.x+)
- CMake (version 3.20+, 3.24+ encouraged)
- Boost program_options library
- NVIDIA GPU with compatible driver

### Install Dependencies (Ubuntu/Debian)
```bash
apt install libboost-program-options-dev
# Or use provided script:
sudo ./debian_install.sh
```

### Build (single-node)
```bash
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth
cmake .
make
```

### Build (multinode)
```bash
cmake -DMULTINODE=1 .
make
```

## CLI Arguments

```
./nvbandwidth -h

nvbandwidth CLI:
  -h [ --help ]                  Produce help message
  -b [ --bufferSize ] arg (=512) Memcpy buffer size in MiB
  -l [ --list ]                  List available testcases
  -t [ --testcase ] arg          Testcase(s) to run (by name or index)
  -p [ --testcasePrefixes ] arg  Testcase(s) to run (by prefix)
  -v [ --verbose ]               Verbose output
  -s [ --skipVerification ]      Skips data verification after copy
  -d [ --disableAffinity ]       Disable automatic CPU affinity control
  -i [ --testSamples ] arg (=3)  Iterations of the benchmark
  -m [ --useMean ]               Use mean instead of median for results
  -j [ --json ]                  Print output in json format instead of plain text.
```

### Hidden arguments (not shown in -h)
- `--loopCount` — Iterations of memcpy within a test sample (default: internal)
- `--perfFormatter` — Use perf formatter prefix (`&&&& PERF`) in output

## Available Testcases (single-node, 35 tests)

### Copy Engine (CE) Tests
| Index | Key | Description |
|-------|-----|-------------|
| 0 | `host_to_device_memcpy_ce` | Host→Device CE memcpy |
| 1 | `device_to_host_memcpy_ce` | Device→Host CE memcpy |
| 2 | `host_to_device_bidirectional_memcpy_ce` | H→D measured while D→H runs simultaneously |
| 3 | `device_to_host_bidirectional_memcpy_ce` | D→H measured while H→D runs simultaneously |
| 4 | `device_to_device_memcpy_read_ce` | D2D read (copy from peer to target using target context) |
| 5 | `device_to_device_memcpy_write_ce` | D2D write (copy from target to peer using target context) |
| 6 | `device_to_device_bidirectional_memcpy_read_ce` | D2D bidir read |
| 7 | `device_to_device_bidirectional_memcpy_write_ce` | D2D bidir write |
| 8 | `all_to_host_memcpy_ce` | All GPUs→Host simultaneously |
| 9 | `all_to_host_bidirectional_memcpy_ce` | All GPUs↔Host bidirectional |
| 10 | `host_to_all_memcpy_ce` | Host→All GPUs simultaneously |
| 11 | `host_to_all_bidirectional_memcpy_ce` | Host↔All GPUs bidirectional |
| 12 | `all_to_one_write_ce` | All peers→One device write |
| 13 | `all_to_one_read_ce` | All peers→One device read |
| 14 | `one_to_all_write_ce` | One device→All peers write |
| 15 | `one_to_all_read_ce` | One device→All peers read |

### Streaming Multiprocessor (SM) Tests
| Index | Key | Description |
|-------|-----|-------------|
| 16 | `host_to_device_memcpy_sm` | Host→Device SM copy kernel |
| 17 | `device_to_host_memcpy_sm` | Device→Host SM copy kernel |
| 18 | `host_to_device_bidirectional_memcpy_sm` | H→D bidir SM |
| 19 | `device_to_host_bidirectional_memcpy_sm` | D→H bidir SM |
| 20 | `device_to_device_memcpy_read_sm` | D2D read SM |
| 21 | `device_to_device_memcpy_write_sm` | D2D write SM |
| 22 | `device_to_device_bidirectional_memcpy_read_sm` | D2D bidir read SM |
| 23 | `device_to_device_bidirectional_memcpy_write_sm` | D2D bidir write SM |
| 24 | `all_to_host_memcpy_sm` | All GPUs→Host SM |
| 25 | `all_to_host_bidirectional_memcpy_sm` | All GPUs↔Host bidir SM |
| 26 | `host_to_all_memcpy_sm` | Host→All GPUs SM |
| 27 | `host_to_all_bidirectional_memcpy_sm` | Host↔All GPUs bidir SM |
| 28 | `all_to_one_write_sm` | All→One write SM |
| 29 | `all_to_one_read_sm` | All→One read SM |
| 30 | `one_to_all_write_sm` | One→All write SM |
| 31 | `one_to_all_read_sm` | One→All read SM |

### Latency & Local Tests
| Index | Key | Description |
|-------|-----|-------------|
| 32 | `host_device_latency_sm` | Host↔Device latency (ns) via ptr chase |
| 33 | `device_to_device_latency_sm` | D2D latency (ns) via ptr chase |
| 34 | `device_local_copy` | Intra-device memcpy bandwidth |

### Multinode Tests (only with MULTINODE=1 build)
- `multinode_device_to_device_memcpy_read_ce`
- `multinode_device_to_device_memcpy_write_ce`
- `multinode_device_to_device_bidirectional_memcpy_read_ce`
- `multinode_device_to_device_bidirectional_memcpy_write_ce`
- `multinode_device_to_device_memcpy_read_sm`
- `multinode_device_to_device_memcpy_write_sm`
- `multinode_device_to_device_bidirectional_memcpy_read_sm`
- `multinode_device_to_device_bidirectional_memcpy_write_sm`
- `multinode_device_to_device_all_to_one_write_sm`
- `multinode_device_to_device_all_from_one_read_sm`
- `multinode_device_to_device_broadcast_one_to_all_sm`
- `multinode_device_to_device_broadcast_all_to_all_sm`
- `multinode_bisect_write_ce`

## Plain Text Output Format

### Unidirectional (1-row matrix)
```
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     26.03     25.94     25.97     26.00     26.19     25.95     26.00     25.97
```

### Peer-to-peer (NxN matrix)
```
Running device_to_device_memcpy_write_ce.
memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0      0.00    276.07    276.36    276.14    276.29    276.48    276.55    276.33
1    276.19      0.00    276.29    276.29    276.57    276.48    276.38    276.24
...
```

## JSON Output Format (--json flag)

The JSON output wraps everything in a root `"nvbandwidth"` object:

```json
{
  "nvbandwidth": {
    "version": "0.8",
    "git_version": "...",
    "CUDA Runtime Version": 12040,
    "Driver Version": "550.54.15",
    "GPU Device list": [
      "0: NVIDIA H100 80GB HBM3 (PCI:0000:07:00.0): (hostname)",
      "1: NVIDIA H100 80GB HBM3 (PCI:0000:0a:00.0): (hostname)"
    ],
    "testcases": [
      {
        "name": "host_to_device_memcpy_ce",
        "status": "Passed",
        "bandwidth_description": "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)",
        "bandwidth_matrix": [
          ["26.03", "25.94", "25.97", "26.00", "26.19", "25.95", "26.00", "25.97"]
        ],
        "sum": 207.88
      },
      {
        "name": "device_to_device_memcpy_write_ce",
        "status": "Passed",
        "bandwidth_description": "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)",
        "bandwidth_matrix": [
          ["0.00", "276.07", "276.36", ...],
          ["276.19", "0.00", "276.29", ...],
          ...
        ],
        "sum": 15467.52
      }
    ]
  }
}
```

### JSON Fields per Testcase
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Testcase key name |
| `status` | string | "Passed", "Running", "Waived", "Error", "Not Found" |
| `bandwidth_description` | string | Human-readable description of the matrix |
| `bandwidth_matrix` | array[array[string]] | 2D array of bandwidth values in GB/s (or "N/A") |
| `sum` | number | Sum of all values in the matrix |
| `error` | string | (optional) Error message if status is "Error" |

### With --verbose flag, additional fields:
| Field | Type | Description |
|-------|------|-------------|
| `min` | number | Minimum value in the matrix |
| `max` | number | Maximum value in the matrix |
| `average` | number | Average of non-zero values |

### Waived Testcases
Tests that can't run (e.g., D2D on single GPU) report `"status": "Waived"` with no bandwidth data.

## Measurement Details

- Default: 3 iterations (`--testSamples`), reports **median** bandwidth
- Can switch to **mean** with `--useMean`
- Default buffer size: 512 MiB (`--bufferSize`)
- Uses spin kernel + CUDA events for precise timing
- Bandwidth reported in **GB/s** (10^9 bytes/sec)
- Latency tests use fixed 2 MiB buffer (ignores --bufferSize)

## Integration Considerations

1. **Running specific tests**: Use `-t <testname>` for one or more tests by name/index. Use `-p <prefix>` for prefix matching (e.g., `-p device_to_device` runs all D2D tests).

2. **JSON parsing**: With `--json`, all output goes to stdout as a single JSON object. The bandwidth matrix values are **strings** (not numbers) in the JSON.

3. **Filtering**: Some tests are auto-skipped ("Waived") if the system doesn't support them (e.g., no peer access for D2D tests on single GPU).

4. **Binary path**: After build, the binary is at `./nvbandwidth` in the build directory.

5. **Latency tests**: `host_device_latency_sm` and `device_to_device_latency_sm` report in **nanoseconds** (ns) rather than GB/s.

6. **Exit code**: Returns 0 on success.

7. **Self-contained**: No external runtime deps after build (statically linked CUDA).
