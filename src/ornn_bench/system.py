"""System environment diagnostics and tool probing.

Provides subprocess-based detection of GPU hardware, drivers, CUDA toolkit,
benchmark tools (nvbandwidth, nccl-tests, mamf-finder), and Python/PyTorch
availability. All probes are designed to work gracefully on non-GPU hosts.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class ToolProbeResult:
    """Result of probing a single external tool."""

    name: str
    available: bool = False
    version: str = ""
    path: str = ""
    error: str = ""
    remediation: str = ""


@dataclass
class GPUProbeResult:
    """Result of probing NVIDIA GPU presence and driver status."""

    detected: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    driver_version: str = ""
    cuda_version: str = ""
    error: str = ""
    remediation: str = ""


@dataclass
class PythonProbeResult:
    """Result of probing Python and PyTorch availability."""

    python_version: str = ""
    pytorch_available: bool = False
    pytorch_version: str = ""
    pytorch_cuda_available: bool = False
    pytorch_cuda_version: str = ""


@dataclass
class EnvironmentInfo:
    """Aggregated environment diagnostics."""

    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""
    cpu_model: str = ""
    gpu: GPUProbeResult = field(default_factory=GPUProbeResult)
    python: PythonProbeResult = field(default_factory=PythonProbeResult)
    tools: list[ToolProbeResult] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        """Return True if an NVIDIA GPU was detected."""
        return self.gpu.detected and self.gpu.gpu_count > 0

    @property
    def all_tools_available(self) -> bool:
        """Return True if all probed tools are available."""
        return all(t.available for t in self.tools)

    @property
    def missing_tools(self) -> list[ToolProbeResult]:
        """Return list of tools that are not available."""
        return [t for t in self.tools if not t.available]


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 10,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Returns (-1, "", error_message) if the process cannot be started.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except OSError as exc:
        return -1, "", f"OS error running {cmd[0]}: {exc}"


def probe_nvidia_smi() -> GPUProbeResult:
    """Probe nvidia-smi for GPU presence and driver information."""
    result = GPUProbeResult()

    # Check if nvidia-smi exists on PATH
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        result.error = "nvidia-smi not found on PATH"
        result.remediation = (
            "Install NVIDIA GPU drivers. "
            "On Ubuntu: sudo apt install nvidia-driver-535. "
            "Verify with: nvidia-smi"
        )
        return result

    # Query GPU count and names
    rc, stdout, stderr = _run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if rc != 0:
        result.error = f"nvidia-smi query failed: {stderr or 'unknown error'}"
        result.remediation = (
            "Ensure NVIDIA drivers are properly installed and a GPU is present. "
            "Check dmesg for driver errors."
        )
        return result

    gpu_names = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not gpu_names:
        result.error = "nvidia-smi returned no GPUs"
        result.remediation = "No NVIDIA GPUs detected. Ensure GPU hardware is installed."
        return result

    result.detected = True
    result.gpu_count = len(gpu_names)
    result.gpu_names = gpu_names

    # Query driver version
    rc, stdout, _ = _run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if rc == 0 and stdout:
        result.driver_version = stdout.splitlines()[0].strip()

    # Query CUDA version from nvidia-smi
    rc, stdout, _ = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv"])
    # CUDA version is typically in the header output; use a different approach
    rc2, full_output, _ = _run_cmd(["nvidia-smi"])
    if rc2 == 0:
        for line in full_output.splitlines():
            if "CUDA Version" in line:
                # Parse "CUDA Version: 12.2" pattern
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    result.cuda_version = parts[1].strip().split()[0].rstrip("|").strip()
                break

    return result


def probe_tool(
    name: str,
    cmd: list[str],
    *,
    version_flag: str = "--version",
    remediation: str = "",
) -> ToolProbeResult:
    """Probe an external tool for availability and version.

    Parameters
    ----------
    name:
        Human-readable tool name.
    cmd:
        Command to run to check availability (e.g. ["nvbandwidth"]).
    version_flag:
        Flag to append to get version info.
    remediation:
        Actionable message if the tool is missing.
    """
    result = ToolProbeResult(name=name)
    tool_path = shutil.which(cmd[0])

    if tool_path is None:
        result.available = False
        result.error = f"{cmd[0]} not found on PATH"
        result.remediation = remediation or f"Install {name} and ensure it is on your PATH."
        return result

    result.path = tool_path

    # Try to get version
    rc, stdout, stderr = _run_cmd([*cmd, version_flag])
    if rc == 0:
        result.available = True
        # Take first non-empty line as version
        version_output = stdout or stderr
        for line in version_output.splitlines():
            if line.strip():
                result.version = line.strip()
                break
        if not result.version:
            result.version = "available"
    else:
        # Tool exists but version flag failed — still consider it available
        result.available = True
        result.version = "unknown"

    return result


def probe_python_environment() -> PythonProbeResult:
    """Probe Python version and PyTorch availability."""
    result = PythonProbeResult()
    result.python_version = platform.python_version()

    try:
        import torch  # type: ignore[import-not-found]

        result.pytorch_available = True
        result.pytorch_version = torch.__version__
        result.pytorch_cuda_available = torch.cuda.is_available()
        if result.pytorch_cuda_available:
            result.pytorch_cuda_version = torch.version.cuda or ""
    except ImportError:
        result.pytorch_available = False

    return result


def _get_cpu_model() -> str:
    """Attempt to get CPU model string."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        rc, stdout, _ = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if rc == 0 and stdout:
            return stdout.strip()

    return platform.processor() or "unknown"


def collect_environment_info() -> EnvironmentInfo:
    """Collect comprehensive environment diagnostics.

    Probes GPU, Python, and benchmark tool availability. Returns an
    :class:`EnvironmentInfo` with all findings, including actionable
    remediation messages for missing components.
    """
    env = EnvironmentInfo()

    # OS info
    env.os_name = f"{platform.system()} {platform.release()}"
    env.os_version = platform.version()
    env.kernel_version = platform.release()
    env.cpu_model = _get_cpu_model()

    # GPU
    env.gpu = probe_nvidia_smi()

    # Python / PyTorch
    env.python = probe_python_environment()

    # Benchmark tools
    env.tools = [
        probe_tool(
            "nvidia-smi",
            ["nvidia-smi"],
            version_flag="--version",
            remediation=(
                "Install NVIDIA GPU drivers. "
                "See: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/"
            ),
        ),
        probe_tool(
            "nvbandwidth",
            ["nvbandwidth"],
            version_flag="--help",
            remediation=(
                "Build nvbandwidth from source: "
                "https://github.com/NVIDIA/nvbandwidth. "
                "Requires CUDA toolkit and CMake."
            ),
        ),
        probe_tool(
            "nccl-tests (all_reduce_perf)",
            ["all_reduce_perf"],
            version_flag="--help",
            remediation=(
                "Build nccl-tests from source: "
                "https://github.com/NVIDIA/nccl-tests. "
                "Requires NCCL library and MPI."
            ),
        ),
    ]

    return env


def check_gpu_available() -> tuple[bool, str]:
    """Quick check if NVIDIA GPU is available for benchmarking.

    Returns
    -------
    tuple[bool, str]
        (is_available, message). If not available, message contains
        actionable remediation guidance.
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return False, (
            "No NVIDIA GPU detected: nvidia-smi not found on PATH.\n\n"
            "To run GPU benchmarks you need:\n"
            "  1. A machine with an NVIDIA GPU\n"
            "  2. NVIDIA drivers installed (nvidia-smi must be available)\n"
            "  3. CUDA toolkit installed\n\n"
            "If you are on a cloud instance, ensure you selected a GPU instance type.\n"
            "For driver installation: "
            "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/\n\n"
            "You can still use 'ornn-bench info' to check your environment\n"
            "or 'ornn-bench report <file>' to view a saved report."
        )

    rc, stdout, stderr = _run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if rc != 0:
        return False, (
            "nvidia-smi is installed but failed to query GPUs.\n\n"
            f"Error: {stderr or 'unknown error'}\n\n"
            "Possible causes:\n"
            "  - No NVIDIA GPU hardware present\n"
            "  - Driver/kernel module mismatch (try rebooting)\n"
            "  - Insufficient permissions (try running as root)\n\n"
            "Check system logs: dmesg | grep -i nvidia"
        )

    gpu_names = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not gpu_names:
        return False, (
            "nvidia-smi returned no GPUs.\n\n"
            "Ensure NVIDIA GPU hardware is present and drivers are loaded.\n"
            "Check: lspci | grep -i nvidia"
        )

    return True, f"Found {len(gpu_names)} NVIDIA GPU(s): {', '.join(gpu_names)}"
