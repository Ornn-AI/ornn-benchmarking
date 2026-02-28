"""Output parsers for benchmark tools.

Provides deterministic parsers for mamf-finder, nvbandwidth, and nccl-tests
output formats. Each parser returns structured dicts suitable for inclusion
in the benchmark report.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# mamf-finder parser
# ---------------------------------------------------------------------------


def parse_mamf_output(raw: str) -> dict[str, Any]:
    """Parse mamf-finder.py text output into structured data.

    Extracts dtype, device, per-shape TFLOPS results, and best result.

    Parameters
    ----------
    raw:
        Raw text output from mamf-finder.py.

    Returns
    -------
    dict with keys: dtype, device, results (list of dicts), best (dict or None)
    """
    result: dict[str, Any] = {
        "dtype": "",
        "device": "",
        "results": [],
        "best": None,
    }

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Parse header: "dtype: bf16 | device: NVIDIA H100 80GB HBM3 | GPU 0"
        dtype_match = re.match(r"dtype:\s*(\S+)", line)
        if dtype_match:
            result["dtype"] = dtype_match.group(1)
            device_match = re.search(r"device:\s*(.+?)(?:\s*\|\s*GPU|\s*$)", line)
            if device_match:
                result["device"] = device_match.group(1).strip()
            gpu_match = re.search(r"GPU\s*(\d+)", line)
            if gpu_match:
                result["gpu_index"] = int(gpu_match.group(1))
            continue

        # Parse data rows: "1024  1024  1024   312.5"
        row_match = re.match(r"(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)", line)
        if row_match:
            result["results"].append({
                "m": int(row_match.group(1)),
                "n": int(row_match.group(2)),
                "k": int(row_match.group(3)),
                "tflops": float(row_match.group(4)),
            })
            continue

        # Parse best line: "Best: M=8192, N=8192, K=8192, 891.3 TFLOPS"
        best_match = re.match(
            r"(?:Best|Result):\s*M=(\d+),\s*N=(\d+),\s*K=(\d+),\s*([\d.]+)\s*TFLOPS",
            line,
        )
        if best_match:
            result["best"] = {
                "m": int(best_match.group(1)),
                "n": int(best_match.group(2)),
                "k": int(best_match.group(3)),
                "tflops": float(best_match.group(4)),
            }

    return result


# ---------------------------------------------------------------------------
# nvbandwidth parser
# ---------------------------------------------------------------------------


def parse_nvbandwidth_json(raw: str) -> dict[str, Any]:
    """Parse nvbandwidth JSON output into structured data.

    Parameters
    ----------
    raw:
        Raw JSON string from nvbandwidth -j output.

    Returns
    -------
    dict with keys: testname, bandwidth_matrix, sum, min, max
    """
    data = json.loads(raw)
    return {
        "testname": data.get("testname", ""),
        "bandwidth_matrix": data.get("bandwidth_matrix", []),
        "sum": float(data.get("sum", 0.0)),
        "min": float(data.get("min", 0.0)),
        "max": float(data.get("max", 0.0)),
    }


# ---------------------------------------------------------------------------
# nccl-tests parser
# ---------------------------------------------------------------------------

# Regex for nccl-tests data lines
_NCCL_DATA_RE = re.compile(
    r"^\s*(\d+)\s+"        # size (bytes)
    r"(\d+)\s+"            # count (elements)
    r"(\w+)\s+"            # type
    r"(\w+)\s+"            # redop
    r"(-?\d+)\s+"          # root
    r"([\d.]+)\s+"         # time (us) out-of-place
    r"([\d.]+)\s+"         # algbw out-of-place
    r"([\d.]+)\s+"         # busbw out-of-place
    r"(\d+)\s+"            # wrong out-of-place
    r"([\d.]+)\s+"         # time (us) in-place
    r"([\d.]+)\s+"         # algbw in-place
    r"([\d.]+)\s+"         # busbw in-place
    r"(\d+)"               # wrong in-place
)

_NCCL_AVG_BW_RE = re.compile(r"#\s*Avg\s+bus\s+bandwidth\s*:\s*([\d.]+)")


def parse_nccl_output(raw: str) -> dict[str, Any]:
    """Parse nccl-tests text output into structured data.

    Extracts device info, per-size bandwidth results, and average bus bandwidth.

    Parameters
    ----------
    raw:
        Raw text output from an nccl-tests binary.

    Returns
    -------
    dict with keys: devices, results (list), avg_bus_bandwidth, max_busbw
    """
    result: dict[str, Any] = {
        "devices": [],
        "results": [],
        "avg_bus_bandwidth": 0.0,
        "max_busbw": 0.0,
    }

    max_busbw = 0.0

    for line in raw.splitlines():
        # Parse device lines: "Rank  0 Group  0 Pid  12345 on hostname device  0 [0x04] NVIDIA H100"
        device_match = re.search(
            r"Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+(\d+)\s+on\s+(\S+)\s+device\s+(\d+)\s+"
            r"\[([^\]]+)\]\s+(.+)",
            line,
        )
        if device_match:
            result["devices"].append({
                "rank": int(device_match.group(1)),
                "group": int(device_match.group(2)),
                "pid": int(device_match.group(3)),
                "hostname": device_match.group(4),
                "device_index": int(device_match.group(5)),
                "pci_id": device_match.group(6),
                "gpu_name": device_match.group(7).strip(),
            })
            continue

        # Parse data rows
        data_match = _NCCL_DATA_RE.match(line)
        if data_match:
            busbw_oop = float(data_match.group(8))
            busbw_ip = float(data_match.group(12))
            max_busbw = max(max_busbw, busbw_oop, busbw_ip)
            result["results"].append({
                "size_bytes": int(data_match.group(1)),
                "count": int(data_match.group(2)),
                "type": data_match.group(3),
                "redop": data_match.group(4),
                "out_of_place": {
                    "time_us": float(data_match.group(6)),
                    "algbw": float(data_match.group(7)),
                    "busbw": busbw_oop,
                    "errors": int(data_match.group(9)),
                },
                "in_place": {
                    "time_us": float(data_match.group(10)),
                    "algbw": float(data_match.group(11)),
                    "busbw": busbw_ip,
                    "errors": int(data_match.group(13)),
                },
            })
            continue

        # Parse avg bus bandwidth
        avg_match = _NCCL_AVG_BW_RE.search(line)
        if avg_match:
            result["avg_bus_bandwidth"] = float(avg_match.group(1))

    result["max_busbw"] = max_busbw
    return result


# ---------------------------------------------------------------------------
# nvidia-smi -q parser (pre-flight inventory)
# ---------------------------------------------------------------------------


def parse_nvidia_smi_q(raw: str) -> dict[str, Any]:
    """Parse nvidia-smi -q output for pre-flight inventory.

    Extracts per-GPU identity info (UUID, name, serial), ECC status,
    NVLink topology, driver/CUDA versions, and memory info.

    Parameters
    ----------
    raw:
        Raw text output from nvidia-smi -q.

    Returns
    -------
    dict with driver/cuda versions and per-gpu details.
    """
    result: dict[str, Any] = {
        "driver_version": "",
        "cuda_version": "",
        "attached_gpus": 0,
        "gpus": [],
    }

    # Parse top-level header
    driver_match = re.search(r"Driver Version\s*:\s*(.+)", raw)
    if driver_match:
        result["driver_version"] = driver_match.group(1).strip()

    cuda_match = re.search(r"CUDA Version\s*:\s*(.+)", raw)
    if cuda_match:
        result["cuda_version"] = cuda_match.group(1).strip()

    attached_match = re.search(r"Attached GPUs\s*:\s*(\d+)", raw)
    if attached_match:
        result["attached_gpus"] = int(attached_match.group(1))

    # Split into per-GPU blocks: "GPU 00000000:XX:00.0"
    gpu_blocks = re.split(r"(?=^GPU\s+[0-9a-fA-F]+:)", raw, flags=re.MULTILINE)

    for block in gpu_blocks:
        if not re.match(r"GPU\s+[0-9a-fA-F]+:", block.strip()):
            continue

        gpu: dict[str, Any] = {
            "pci_bus_id": "",
            "product_name": "",
            "uuid": "",
            "serial_number": "",
            "memory_total_mib": 0,
            "ecc_mode": "",
            "ecc_errors": {},
            "nvlink": [],
            "temperature_gpu_c": 0,
            "power_draw_w": 0.0,
            "power_limit_w": 0.0,
        }

        # PCI bus ID
        pci_match = re.match(r"GPU\s+([0-9a-fA-F:\.]+)", block.strip())
        if pci_match:
            gpu["pci_bus_id"] = pci_match.group(1)

        _extract_field(block, "Product Name", gpu, "product_name")
        _extract_field(block, "GPU UUID", gpu, "uuid")
        _extract_field(block, "Serial Number", gpu, "serial_number")

        # Memory total
        mem_match = re.search(r"Total\s*:\s*(\d+)\s*MiB", block)
        if mem_match:
            gpu["memory_total_mib"] = int(mem_match.group(1))

        # Temperature
        temp_match = re.search(r"GPU Current Temp\s*:\s*(\d+)\s*C", block)
        if temp_match:
            gpu["temperature_gpu_c"] = int(temp_match.group(1))

        # Power
        power_match = re.search(r"Power Draw\s*:\s*([\d.]+)\s*W", block)
        if power_match:
            gpu["power_draw_w"] = float(power_match.group(1))

        limit_match = re.search(r"Current Power Limit\s*:\s*([\d.]+)\s*W", block)
        if limit_match:
            gpu["power_limit_w"] = float(limit_match.group(1))

        # ECC mode
        ecc_match = re.search(r"ECC Mode\s*\n\s*Current\s*:\s*(\w+)", block)
        if ecc_match:
            gpu["ecc_mode"] = ecc_match.group(1)

        # ECC errors (volatile)
        gpu["ecc_errors"] = _parse_ecc_errors(block)

        # NVLink
        gpu["nvlink"] = _parse_nvlink_entries(block)

        result["gpus"].append(gpu)

    return result


def _extract_field(
    block: str, field_name: str, target: dict[str, Any], key: str
) -> None:
    """Extract a simple 'Field : Value' from nvidia-smi output."""
    match = re.search(rf"{re.escape(field_name)}\s*:\s*(.+)", block)
    if match:
        target[key] = match.group(1).strip()


def _parse_ecc_errors(block: str) -> dict[str, int]:
    """Parse ECC error counters from an nvidia-smi GPU block."""
    errors: dict[str, int] = {}
    # Match volatile section
    volatile_match = re.search(
        r"Volatile\s*\n((?:\s+\S+\s+(?:Correctable|Uncorrectable)\s*:\s*\d+\s*\n?)+)",
        block,
    )
    if volatile_match:
        for m in re.finditer(
            r"(\S+)\s+(Correctable|Uncorrectable)\s*:\s*(\d+)",
            volatile_match.group(1),
        ):
            key = f"volatile_{m.group(1).lower()}_{m.group(2).lower()}"
            errors[key] = int(m.group(3))

    return errors


def _parse_nvlink_entries(block: str) -> list[dict[str, str]]:
    """Parse NVLink entries from an nvidia-smi GPU block."""
    links: list[dict[str, str]] = []
    # Find NvLink section
    nvlink_section = re.search(r"NvLink\s*\n((?:\s+.+\n)*)", block)
    if not nvlink_section:
        return links

    section_text = nvlink_section.group(1)
    link_blocks = re.split(r"(?=\s+Link\s+\d+)", section_text)

    for lb in link_blocks:
        link_match = re.match(r"\s+Link\s+(\d+)", lb)
        if not link_match:
            continue

        link: dict[str, str] = {"link_id": link_match.group(1)}

        state_match = re.search(r"State\s*:\s*(\w+)", lb)
        if state_match:
            link["state"] = state_match.group(1)

        remote_match = re.search(r"Remote GPU UUID\s*:\s*(\S+)", lb)
        if remote_match:
            link["remote_gpu_uuid"] = remote_match.group(1)

        links.append(link)

    return links
