from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_NODE_SCRIPT = ROOT / "ornn-bench-infra" / "benchmark_node.sh"

ORNN_BENCH_VERSION = "0.2.0"
MAMF_FINDER_COMMIT = "7c660da71e533fdb5de141591379d3c8070ef272"
NVBANDWIDTH_HEAD = "66746a3bef61c8c2e12ab34955310da70b9e38cb"
NCCL_TESTS_COMMIT = "ae98985f5599617be94042f4aa3637d10014ce89"
NCCL_DEV_VERSION = "2.21.5-1+cuda12.4"


@dataclass
class StubEnvironment:
    env: dict[str, str]
    output_root: Path
    invocation_log: Path
    campaign_dir: Path


def _write_file(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _build_topology(link_token: str) -> str:
    header = "        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 CPU Affinity"
    rows = [header]
    for gpu_index in range(8):
        tokens = ["X" if peer == gpu_index else link_token for peer in range(8)]
        rows.append(f"GPU{gpu_index}    {' '.join(tokens)} 0-95")
    return "\n".join(rows)


def _parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        parsed[key] = value.strip().strip('"')
    return parsed


def _collect_file_texts(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _create_stub_environment(
    tmp_path: Path,
    *,
    campaign_id: str,
    ornn_version: str = ORNN_BENCH_VERSION,
    gpu_count: int = 8,
    memory_mib: int = 81920,
    topology_token: str = "NV18",
    default_output: bool = False,
    upload_mode: str = "auto",
    api_key: str | None = None,
) -> StubEnvironment:
    tools_root = tmp_path / "tools"
    src_root = tools_root / "src"
    bin_root = tools_root / "bin"
    venv_bin = tools_root / "venv" / "bin"
    stub_bin = tmp_path / "stub-bin"
    output_root = tmp_path / "output"
    invocation_log = tmp_path / "ornn-bench-invocations.log"
    runtime_env = tools_root / "runtime.env"
    provider = "aws"
    region = "us-central1"
    instance_type = "h100-sxm5"
    campaign_dir = output_root / provider / region / instance_type / campaign_id

    mamf_repo_dir = src_root / "ml-engineering"
    mamf_path = mamf_repo_dir / "compute" / "accelerator" / "benchmarks" / "mamf-finder.py"
    nvbandwidth_repo_dir = src_root / "nvbandwidth"
    nccl_tests_repo_dir = src_root / "nccl-tests"
    cuda_home = tools_root / "cuda"
    cuda_bin_dir = cuda_home / "bin"
    cuda_lib_dir = cuda_home / "lib64"
    nccl_home = tools_root / "nccl"
    nccl_lib_dir = nccl_home / "lib64"

    for directory in (
        nvbandwidth_repo_dir,
        nccl_tests_repo_dir,
        cuda_home,
        cuda_bin_dir,
        cuda_lib_dir,
        nccl_home,
        nccl_lib_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _write_file(mamf_path, "#!/usr/bin/env python3\n", executable=True)
    for path in (
        bin_root / "nvbandwidth",
        bin_root / "all_reduce_perf",
        bin_root / "reduce_scatter_perf",
        bin_root / "all_gather_perf",
        bin_root / "broadcast_perf",
        bin_root / "sendrecv_perf",
    ):
        _write_file(path, "#!/usr/bin/env bash\nexit 0\n", executable=True)

    _write_file(
        cuda_bin_dir / "nvcc",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf 'Cuda compilation tools, release 12.4, V12.4.131\n'
            """
        ),
        executable=True,
    )

    _write_file(
        venv_bin / "python",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            if [[ "${{1:-}}" == "-c" ]]; then
              printf '3.10\n'
              exit 0
            fi

            if [[ "${{1:-}}" == "-" && "${{2:-}}" == "2.5.1" ]]; then
              cat >/dev/null
              printf '2.5.1+cu124\n12.4\n'
              exit 0
            fi

            exec "{sys.executable}" "$@"
            """
        ),
        executable=True,
    )

    _write_file(
        bin_root / "ornn-bench",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail

            if [[ "${1:-}" == "--version" ]]; then
              printf 'ornn-bench %s\n' "${STUB_ORNN_VERSION:-0.1.0}"
              exit 0
            fi

            if [[ "${1:-}" != "run" ]]; then
              printf 'unsupported invocation: %s\n' "$*" >&2
              exit 2
            fi

            shift
            output_path=""
            upload_enabled=0
            while [[ "$#" -gt 0 ]]; do
              case "$1" in
                -o|--output)
                  output_path="$2"
                  shift 2
                  ;;
                --upload)
                  upload_enabled=1
                  shift
                  ;;
                *)
                  shift
                  ;;
              esac
            done

            printf '%s|upload=%s\n' "${output_path}" "${upload_enabled}" >> "${STUB_INVOCATION_LOG}"

            report_payload='{"scores":{"ornn_i":95.0,"ornn_t":94.0,"status":"complete","components":{"bw":1.0,"fp8":1.0,"bf16":1.0,"ar":1.0}},"manifest":{"artifacts":["stub"]}}'
            if [[ "${STUB_DEFAULT_OUTPUT:-0}" == "1" ]]; then
              default_path="$(pwd)/ornn_report_stub_$(basename "${output_path}")"
              printf '%s\n' "${report_payload}" > "${default_path}"
            else
              mkdir -p "$(dirname "${output_path}")"
              printf '%s\n' "${report_payload}" > "${output_path}"
            fi

            exit "${STUB_RUN_EXIT_CODE:-0}"
            """
        ),
        executable=True,
    )

    topology = _build_topology(topology_token)
    gpu_lines = "\n".join(
        f"NVIDIA H100 80GB HBM3, {memory_mib}, 550.54.15" for _ in range(gpu_count)
    )
    nvidia_smi_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'if [[ "${1:-}" == "--query-gpu=driver_version" ]]; then',
        "  printf '550.54.15\\n'",
        "  exit 0",
        "fi",
        "",
        'if [[ "${1:-}" == "--query-gpu=name,memory.total,driver_version" ]]; then',
        "  cat <<'EOF'",
        *gpu_lines.splitlines(),
        "EOF",
        "  exit 0",
        "fi",
        "",
        'if [[ "${1:-}" == "topo" && "${2:-}" == "-m" ]]; then',
        "  cat <<'EOF'",
        *topology.splitlines(),
        "EOF",
        "  exit 0",
        "fi",
        "",
        "printf 'unexpected nvidia-smi args: %s\\n' \"$*\" >&2",
        "exit 2",
        "",
    ]
    _write_file(stub_bin / "nvidia-smi", "\n".join(nvidia_smi_lines), executable=True)

    _write_file(
        stub_bin / "git",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            repo_name="$(basename "$2")"
            if [[ "$3" == "rev-parse" && "$4" == "HEAD" ]]; then
              case "${{repo_name}}" in
                ml-engineering)
                  printf '{MAMF_FINDER_COMMIT}\n'
                  ;;
                nvbandwidth)
                  printf '{NVBANDWIDTH_HEAD}\n'
                  ;;
                nccl-tests)
                  printf '{NCCL_TESTS_COMMIT}\n'
                  ;;
                *)
                  exit 1
                  ;;
              esac
              exit 0
            fi

            if [[ "$3" == "tag" && "$4" == "--points-at" && "$5" == "HEAD" ]]; then
              if [[ "${{repo_name}}" == "nvbandwidth" ]]; then
                printf 'v0.8\n'
              fi
              exit 0
            fi

            printf 'unexpected git args: %s\n' "$*" >&2
            exit 2
            """
        ),
        executable=True,
    )

    _write_file(
        stub_bin / "dpkg-query",
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            package_name="${{@:$#}}"
            case "${{package_name}}" in
              libnccl2|libnccl-dev)
                printf '{NCCL_DEV_VERSION}\n'
                ;;
              *)
                exit 1
                ;;
            esac
            """
        ),
        executable=True,
    )

    runtime_env.write_text(
        "\n".join(
            [
                f'ORNN_BENCH_TOOLS_ROOT="{tools_root}"',
                f'ORNN_BENCH_VENV="{tools_root / "venv"}"',
                f'ORNN_BENCH_PYTHON="{venv_bin / "python"}"',
                f'TOOLS_BIN_DIR="{bin_root}"',
                f'ORNN_BENCH_BIN="{bin_root / "ornn-bench"}"',
                f'MAMF_FINDER_REPO_DIR="{mamf_repo_dir}"',
                f'MAMF_FINDER_PATH="{mamf_path}"',
                f'NVBANDWIDTH_REPO_DIR="{nvbandwidth_repo_dir}"',
                f'NVBANDWIDTH_BUILD_DIR="{nvbandwidth_repo_dir / "build"}"',
                f'NVBANDWIDTH_PATH="{bin_root / "nvbandwidth"}"',
                f'NCCL_TESTS_REPO_DIR="{nccl_tests_repo_dir}"',
                f'NCCL_TESTS_BIN_DIR="{nccl_tests_repo_dir / "build"}"',
                f'ALL_REDUCE_PERF_PATH="{bin_root / "all_reduce_perf"}"',
                f'REDUCE_SCATTER_PERF_PATH="{bin_root / "reduce_scatter_perf"}"',
                f'ALL_GATHER_PERF_PATH="{bin_root / "all_gather_perf"}"',
                f'BROADCAST_PERF_PATH="{bin_root / "broadcast_perf"}"',
                f'SENDRECV_PERF_PATH="{bin_root / "sendrecv_perf"}"',
                f'CUDA_HOME="{cuda_home}"',
                f'CUDA_BIN_DIR="{cuda_bin_dir}"',
                f'CUDA_LIB_DIR="{cuda_lib_dir}"',
                f'NCCL_HOME="{nccl_home}"',
                f'NCCL_LIB_DIR="{nccl_lib_dir}"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{stub_bin}:{env.get('PATH', '')}",
            "ORNN_BENCH_RUNTIME_ENV": str(runtime_env),
            "ORNN_BENCH_OUTPUT_ROOT": str(output_root),
            "ORNN_BENCH_CAMPAIGN_ID": campaign_id,
            "ORNN_BENCH_COOLDOWN_SECONDS": "0",
            "ORNN_BENCH_UPLOAD": upload_mode,
            "STUB_ORNN_VERSION": ornn_version,
            "STUB_DEFAULT_OUTPUT": "1" if default_output else "0",
            "STUB_INVOCATION_LOG": str(invocation_log),
        }
    )
    if api_key is None:
        env.pop("ORNN_API_KEY", None)
    else:
        env["ORNN_API_KEY"] = api_key

    return StubEnvironment(
        env=env,
        output_root=output_root,
        invocation_log=invocation_log,
        campaign_dir=campaign_dir,
    )


def _run_benchmark_node(stub_env: StubEnvironment) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BENCHMARK_NODE_SCRIPT), "aws", "us-central1", "h100-sxm5"],
        capture_output=True,
        text=True,
        check=False,
        env=stub_env.env,
    )


class TestBenchmarkNodeRunnerExecution:
    def test_completes_three_runs_recovers_default_output_and_preserves_secret(self, tmp_path: Path) -> None:
        api_key = "secret-benchmark-key"
        stub_env = _create_stub_environment(
            tmp_path,
            campaign_id="success-with-upload",
            default_output=True,
            upload_mode="auto",
            api_key=api_key,
        )

        result = _run_benchmark_node(stub_env)

        assert result.returncode == 0, result.stderr
        assert stub_env.campaign_dir.is_dir()

        campaign_manifest = _parse_env_file(stub_env.campaign_dir / "manifests" / "campaign.env")
        assert campaign_manifest["STATUS"] == "completed"
        assert campaign_manifest["COMPLETED_RUNS"] == "3"
        assert campaign_manifest["UPLOAD_ENABLED"] == "1"
        assert campaign_manifest["ORNN_API_KEY_PRESENT"] == "1"

        toolchain_manifest = _parse_env_file(
            stub_env.campaign_dir / "manifests" / "toolchain-verification.env"
        )
        assert toolchain_manifest["ORNN_BENCH_ACTUAL_VERSION"] == ORNN_BENCH_VERSION
        assert toolchain_manifest["MAMF_FINDER_ACTUAL_COMMIT"] == MAMF_FINDER_COMMIT
        assert toolchain_manifest["NCCL_TESTS_ACTUAL_COMMIT"] == NCCL_TESTS_COMMIT

        for run_number in range(1, 4):
            report_path = stub_env.campaign_dir / "reports" / f"run-0{run_number}.json"
            log_path = stub_env.campaign_dir / "logs" / f"run-0{run_number}.log"
            run_manifest = _parse_env_file(
                stub_env.campaign_dir / "manifests" / f"run-0{run_number}.env"
            )

            assert report_path.exists()
            assert log_path.exists()
            assert run_manifest["RECOVERED_FROM_DEFAULT"] == "1"
            assert run_manifest["STATUS"] == "completed"
            assert run_manifest["UPLOAD_ENABLED"] == "1"

        invocation_lines = stub_env.invocation_log.read_text(encoding="utf-8").splitlines()
        assert len(invocation_lines) == 3
        assert all(line.endswith("upload=1") for line in invocation_lines)

        artifact_text = _collect_file_texts(stub_env.campaign_dir)
        assert api_key not in artifact_text

    def test_skips_upload_when_api_key_is_not_set(self, tmp_path: Path) -> None:
        stub_env = _create_stub_environment(
            tmp_path,
            campaign_id="success-without-upload",
            upload_mode="auto",
            api_key=None,
        )

        result = _run_benchmark_node(stub_env)

        assert result.returncode == 0, result.stderr
        campaign_manifest = _parse_env_file(stub_env.campaign_dir / "manifests" / "campaign.env")
        assert campaign_manifest["UPLOAD_ENABLED"] == "0"
        assert campaign_manifest["ORNN_API_KEY_PRESENT"] == "0"

        invocation_lines = stub_env.invocation_log.read_text(encoding="utf-8").splitlines()
        assert len(invocation_lines) == 3
        assert all(line.endswith("upload=0") for line in invocation_lines)

    def test_aborts_on_toolchain_mismatch(self, tmp_path: Path) -> None:
        stub_env = _create_stub_environment(
            tmp_path,
            campaign_id="toolchain-mismatch",
            ornn_version="9.9.9",
            upload_mode="never",
        )

        result = _run_benchmark_node(stub_env)

        assert result.returncode != 0
        assert "ornn-bench version mismatch" in result.stderr

        campaign_manifest = _parse_env_file(stub_env.campaign_dir / "manifests" / "campaign.env")
        assert campaign_manifest["STATUS"] == "failed"
        assert campaign_manifest["COMPLETED_RUNS"] == "0"
        assert not list((stub_env.campaign_dir / "reports").glob("*.json"))

    def test_aborts_when_sxm5_topology_gate_fails_and_prints_topology(self, tmp_path: Path) -> None:
        stub_env = _create_stub_environment(
            tmp_path,
            campaign_id="topology-failure",
            topology_token="SYS",
            upload_mode="never",
        )

        result = _run_benchmark_node(stub_env)

        assert result.returncode != 0
        assert "nvidia-smi topo -m output:" in result.stderr
        assert "SYS" in result.stderr
        assert "topology appears PCIe-only or incomplete" in result.stderr
