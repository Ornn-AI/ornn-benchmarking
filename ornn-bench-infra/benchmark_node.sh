#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly TOOLCHAIN_ENV="${SCRIPT_DIR}/toolchain.env"
readonly DEFAULT_TOOLS_ROOT="/opt/ornn-bench-tools"
readonly TOOLS_ROOT="${ORNN_BENCH_TOOLS_ROOT:-${DEFAULT_TOOLS_ROOT}}"
readonly RUNTIME_ENV="${ORNN_BENCH_RUNTIME_ENV:-${TOOLS_ROOT}/runtime.env}"
readonly OUTPUT_ROOT="${ORNN_BENCH_OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
readonly EXPECTED_GPU_COUNT=8
readonly MEMORY_FLOOR_MIB=79000
readonly RUN_COUNT=3
readonly COOLDOWN_SECONDS="${ORNN_BENCH_COOLDOWN_SECONDS:-60}"
readonly UPLOAD_SETTING="${ORNN_BENCH_UPLOAD:-auto}"

PROVIDER=""
REGION=""
INSTANCE_TYPE=""
CAMPAIGN_ID=""
CAMPAIGN_DIR=""
REPORTS_DIR=""
LOGS_DIR=""
MANIFESTS_DIR=""
WORK_DIR=""
CAMPAIGN_MANIFEST_PATH=""
TOOLCHAIN_MANIFEST_PATH=""
SXM5_MANIFEST_PATH=""
TOPOLOGY_FILE_PATH=""
GPU_INVENTORY_PATH=""
CAMPAIGN_INITIALIZED=0
CAMPAIGN_STATUS="starting"
UPLOAD_ENABLED=0
COMPLETED_RUNS=0
FAIL_REASON=""

ORNN_BENCH_BIN=""
ORNN_BENCH_PYTHON=""
TOOLS_BIN_DIR=""
MAMF_FINDER_REPO_DIR=""
MAMF_FINDER_PATH=""
NVBANDWIDTH_REPO_DIR=""
NVBANDWIDTH_PATH=""
NCCL_TESTS_REPO_DIR=""
ALL_REDUCE_PERF_PATH=""
REDUCE_SCATTER_PERF_PATH=""
ALL_GATHER_PERF_PATH=""
BROADCAST_PERF_PATH=""
SENDRECV_PERF_PATH=""
CUDA_HOME=""
CUDA_BIN_DIR=""
CUDA_LIB_DIR=""
NCCL_HOME=""
NCCL_LIB_DIR=""

ACTUAL_ORNN_BENCH_VERSION=""
ACTUAL_PYTHON_VERSION=""
ACTUAL_TORCH_VERSION=""
ACTUAL_TORCH_CUDA=""
ACTUAL_CUDA_VERSION=""
ACTUAL_DRIVER_VERSION=""
ACTUAL_MAMF_COMMIT=""
ACTUAL_MAMF_PATH=""
ACTUAL_NVBANDWIDTH_HEAD=""
ACTUAL_NVBANDWIDTH_TAG=""
ACTUAL_NCCL_TESTS_COMMIT=""
ACTUAL_LIBNCCL2_VERSION=""
ACTUAL_LIBNCCL_DEV_VERSION=""

GPU_COUNT=0
GPU_NAME_1=""
GPU_NAME_2=""
GPU_NAME_3=""
GPU_NAME_4=""
GPU_NAME_5=""
GPU_NAME_6=""
GPU_NAME_7=""
GPU_NAME_8=""
GPU_MEMORY_1=""
GPU_MEMORY_2=""
GPU_MEMORY_3=""
GPU_MEMORY_4=""
GPU_MEMORY_5=""
GPU_MEMORY_6=""
GPU_MEMORY_7=""
GPU_MEMORY_8=""
TOPOLOGY_OUTPUT=""
TOPOLOGY_GPU_ROW_COUNT=0
TOPOLOGY_ROWS_WITH_NV=0

RUN_1_REPORT=""
RUN_2_REPORT=""
RUN_3_REPORT=""
RUN_1_LOG=""
RUN_2_LOG=""
RUN_3_LOG=""

log() {
  printf '==> %s\n' "$*" >&2
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

write_env_entry() {
  local key="$1"
  local value="$2"

  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s="%s"\n' "${key}" "${value}"
}

write_campaign_manifest() {
  local run_number
  local report_var
  local log_var

  [[ -n "${CAMPAIGN_MANIFEST_PATH}" ]] || return 0
  {
    write_env_entry "STATUS" "${CAMPAIGN_STATUS}"
    write_env_entry "FAIL_REASON" "${FAIL_REASON}"
    write_env_entry "PROVIDER" "${PROVIDER}"
    write_env_entry "REGION" "${REGION}"
    write_env_entry "INSTANCE_TYPE" "${INSTANCE_TYPE}"
    write_env_entry "CAMPAIGN_ID" "${CAMPAIGN_ID}"
    write_env_entry "CAMPAIGN_DIR" "${CAMPAIGN_DIR}"
    write_env_entry "REPORTS_DIR" "${REPORTS_DIR}"
    write_env_entry "LOGS_DIR" "${LOGS_DIR}"
    write_env_entry "MANIFESTS_DIR" "${MANIFESTS_DIR}"
    write_env_entry "WORK_DIR" "${WORK_DIR}"
    write_env_entry "TOOLCHAIN_ENV" "${TOOLCHAIN_ENV}"
    write_env_entry "RUNTIME_ENV" "${RUNTIME_ENV}"
    write_env_entry "RUN_COUNT" "${RUN_COUNT}"
    write_env_entry "COMPLETED_RUNS" "${COMPLETED_RUNS}"
    write_env_entry "COOLDOWN_SECONDS" "${COOLDOWN_SECONDS}"
    write_env_entry "UPLOAD_SETTING" "${UPLOAD_SETTING}"
    write_env_entry "UPLOAD_ENABLED" "${UPLOAD_ENABLED}"
    write_env_entry "ORNN_API_KEY_PRESENT" "$(if [[ -n "${ORNN_API_KEY:-}" ]]; then printf '1'; else printf '0'; fi)"
    write_env_entry "GPU_COUNT" "${GPU_COUNT}"
    write_env_entry "DRIVER_VERSION" "${ACTUAL_DRIVER_VERSION}"
    write_env_entry "TOPOLOGY_FILE" "${TOPOLOGY_FILE_PATH}"
    write_env_entry "GPU_INVENTORY_FILE" "${GPU_INVENTORY_PATH}"
    for run_number in 1 2 3; do
      report_var="RUN_${run_number}_REPORT"
      log_var="RUN_${run_number}_LOG"
      write_env_entry "${report_var}" "${!report_var}"
      write_env_entry "${log_var}" "${!log_var}"
    done
  } > "${CAMPAIGN_MANIFEST_PATH}"
}

fail() {
  FAIL_REASON="$*"
  if [[ "${CAMPAIGN_INITIALIZED}" -eq 1 ]]; then
    CAMPAIGN_STATUS="failed"
    write_campaign_manifest
  fi
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
Usage: benchmark_node.sh <provider> <region> <instance_type>

Runs three OrnnBench captures on a provisioned 8xH100 SXM5 80GB node,
verifies the locked toolchain, enforces the SXM5 topology gate, and stores
reports/logs/manifests under a structured campaign directory.

Optional environment variables:
  ORNN_BENCH_OUTPUT_ROOT        Root directory for campaign artifacts.
  ORNN_BENCH_COOLDOWN_SECONDS   Sleep interval between runs (default: 60).
  ORNN_BENCH_UPLOAD             auto|always|never (default: auto).
  ORNN_BENCH_CAMPAIGN_ID        Override campaign directory suffix.
  ORNN_BENCH_TOOLS_ROOT         Override /opt/ornn-bench-tools for testing.
  ORNN_BENCH_RUNTIME_ENV        Override runtime.env location for testing.

Upload behavior:
  - auto   : enable --upload only when ORNN_API_KEY is present in the env.
  - always : require ORNN_API_KEY and force --upload.
  - never  : never pass --upload.
EOF
  exit 1
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

slugify() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]._-' '-')"
  value="$(trim "${value}")"
  while [[ "${value}" == *--* ]]; do
    value="${value//--/-}"
  done
  value="${value#-}"
  value="${value%-}"
  [[ -n "${value}" ]] || fail "Unable to derive a safe directory name from '$1'."
  printf '%s' "${value}"
}

version_at_least() {
  local lhs="$1"
  local rhs="$2"
  local lhs_parts=()
  local rhs_parts=()
  local i
  local lhs_part
  local rhs_part
  local count
  local IFS=.

  read -r -a lhs_parts <<< "${lhs}"
  read -r -a rhs_parts <<< "${rhs}"

  count="${#lhs_parts[@]}"
  if [[ "${#rhs_parts[@]}" -gt "${count}" ]]; then
    count="${#rhs_parts[@]}"
  fi

  for ((i = 0; i < count; i += 1)); do
    lhs_part="${lhs_parts[i]:-0}"
    rhs_part="${rhs_parts[i]:-0}"
    if ((10#${lhs_part} > 10#${rhs_part})); then
      return 0
    fi
    if ((10#${lhs_part} < 10#${rhs_part})); then
      return 1
    fi
  done

  return 0
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" ]] || fail "${label} not found at ${path}."
}

require_directory() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || fail "${label} not found at ${path}."
}

require_executable() {
  local path="$1"
  local label="$2"
  [[ -x "${path}" ]] || fail "${label} is not executable at ${path}."
}

resolve_upload_mode() {
  case "${UPLOAD_SETTING}" in
    auto|AUTO|Auto)
      if [[ -n "${ORNN_API_KEY:-}" ]]; then
        UPLOAD_ENABLED=1
      else
        UPLOAD_ENABLED=0
      fi
      ;;
    always|ALWAYS|Always|1|true|TRUE|yes|YES|on|ON)
      [[ -n "${ORNN_API_KEY:-}" ]] || fail \
        "ORNN_BENCH_UPLOAD=${UPLOAD_SETTING} requires ORNN_API_KEY to be set in the environment."
      UPLOAD_ENABLED=1
      ;;
    never|NEVER|Never|0|false|FALSE|no|NO|off|OFF)
      UPLOAD_ENABLED=0
      ;;
    *)
      fail "Unsupported ORNN_BENCH_UPLOAD value '${UPLOAD_SETTING}'. Use auto, always, or never."
      ;;
  esac
}

initialize_campaign_layout() {
  local provider_slug
  local region_slug
  local instance_slug
  local requested_campaign_id

  provider_slug="$(slugify "${PROVIDER}")"
  region_slug="$(slugify "${REGION}")"
  instance_slug="$(slugify "${INSTANCE_TYPE}")"
  requested_campaign_id="${ORNN_BENCH_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  CAMPAIGN_ID="$(slugify "${requested_campaign_id}")"

  CAMPAIGN_DIR="${OUTPUT_ROOT}/${provider_slug}/${region_slug}/${instance_slug}/${CAMPAIGN_ID}"
  REPORTS_DIR="${CAMPAIGN_DIR}/reports"
  LOGS_DIR="${CAMPAIGN_DIR}/logs"
  MANIFESTS_DIR="${CAMPAIGN_DIR}/manifests"
  WORK_DIR="${CAMPAIGN_DIR}/work"
  CAMPAIGN_MANIFEST_PATH="${MANIFESTS_DIR}/campaign.env"
  TOOLCHAIN_MANIFEST_PATH="${MANIFESTS_DIR}/toolchain-verification.env"
  SXM5_MANIFEST_PATH="${MANIFESTS_DIR}/sxm5-gate.env"
  TOPOLOGY_FILE_PATH="${MANIFESTS_DIR}/topology.txt"
  GPU_INVENTORY_PATH="${MANIFESTS_DIR}/gpu-inventory.csv"

  mkdir -p "${REPORTS_DIR}" "${LOGS_DIR}" "${MANIFESTS_DIR}" "${WORK_DIR}"
  CAMPAIGN_INITIALIZED=1
  write_campaign_manifest
}

source_locked_env() {
  require_file "${TOOLCHAIN_ENV}" "toolchain lock"
  require_file "${RUNTIME_ENV}" "runtime environment"

  # shellcheck disable=SC1091
  source "${TOOLCHAIN_ENV}"
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV}"

  ORNN_BENCH_BIN="${ORNN_BENCH_BIN:-${TOOLS_BIN_DIR:-}/ornn-bench}"
  ORNN_BENCH_PYTHON="${ORNN_BENCH_PYTHON:-$(command -v "python${PYTHON_VERSION}" 2>/dev/null || true)}"
  TOOLS_BIN_DIR="${TOOLS_BIN_DIR:-$(dirname "${ORNN_BENCH_BIN}")}"
  TOOLS_BIN_DIR="$(trim "${TOOLS_BIN_DIR}")"
}

prepare_runtime_paths() {
  require_executable "${ORNN_BENCH_BIN}" "ornn-bench CLI"
  require_executable "${ORNN_BENCH_PYTHON}" "provisioned Python"
  require_directory "${MAMF_FINDER_REPO_DIR}" "mamf-finder repo"
  require_file "${MAMF_FINDER_PATH}" "mamf-finder script"
  require_directory "${NVBANDWIDTH_REPO_DIR}" "nvbandwidth repo"
  require_executable "${NVBANDWIDTH_PATH}" "nvbandwidth binary"
  require_directory "${NCCL_TESTS_REPO_DIR}" "nccl-tests repo"
  require_executable "${ALL_REDUCE_PERF_PATH}" "all_reduce_perf binary"
  require_executable "${REDUCE_SCATTER_PERF_PATH}" "reduce_scatter_perf binary"
  require_executable "${ALL_GATHER_PERF_PATH}" "all_gather_perf binary"
  require_executable "${BROADCAST_PERF_PATH}" "broadcast_perf binary"
  require_executable "${SENDRECV_PERF_PATH}" "sendrecv_perf binary"
  require_directory "${CUDA_HOME}" "CUDA toolkit"
  require_directory "${CUDA_BIN_DIR}" "CUDA bin directory"
  require_directory "${CUDA_LIB_DIR}" "CUDA library directory"
  require_directory "${NCCL_LIB_DIR}" "NCCL library directory"

  export PATH="${TOOLS_BIN_DIR}:${CUDA_BIN_DIR}:${PATH}"
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${NCCL_LIB_DIR}"
  fi
}

verify_python_pin() {
  ACTUAL_PYTHON_VERSION="$("${ORNN_BENCH_PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" || fail \
    "Unable to query the provisioned Python version via ${ORNN_BENCH_PYTHON}."
  [[ "${ACTUAL_PYTHON_VERSION}" == "${PYTHON_VERSION}" ]] || fail \
    "Python version mismatch: expected ${PYTHON_VERSION}, found ${ACTUAL_PYTHON_VERSION}."
}

verify_ornn_bench_pin() {
  local version_output
  version_output="$("${ORNN_BENCH_BIN}" --version)" || fail \
    "Unable to read ornn-bench version from ${ORNN_BENCH_BIN}."
  ACTUAL_ORNN_BENCH_VERSION="${version_output##* }"
  [[ "${ACTUAL_ORNN_BENCH_VERSION}" == "${ORNN_BENCH_VERSION}" ]] || fail \
    "ornn-bench version mismatch: expected ${ORNN_BENCH_VERSION}, found ${ACTUAL_ORNN_BENCH_VERSION}."
}

verify_torch_pin() {
  local torch_output
  torch_output="$("${ORNN_BENCH_PYTHON}" - "${TORCH_VERSION}" "${TORCH_CUDA_WHEEL}" "${CUDA_VERSION}" <<'PY'
import sys
import torch

expected_version, expected_wheel, expected_cuda = sys.argv[1:4]
print(torch.__version__)
print(torch.version.cuda or "")

actual_base = torch.__version__.split("+", 1)[0]
if actual_base != expected_version:
    raise SystemExit(
        f"torch version mismatch: expected {expected_version}, found {torch.__version__}"
    )
if f"+{expected_wheel}" not in torch.__version__:
    raise SystemExit(
        f"torch wheel mismatch: expected +{expected_wheel}, found {torch.__version__}"
    )
if (torch.version.cuda or "") != expected_cuda:
    raise SystemExit(
        f"torch CUDA mismatch: expected {expected_cuda}, found {torch.version.cuda}"
    )
PY
)" || fail "Unable to verify the provisioned torch installation."

  ACTUAL_TORCH_VERSION="$(printf '%s\n' "${torch_output}" | sed -n '1p')"
  ACTUAL_TORCH_CUDA="$(printf '%s\n' "${torch_output}" | sed -n '2p')"
}

verify_cuda_pin() {
  ACTUAL_CUDA_VERSION="$("${CUDA_BIN_DIR}/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')" || fail \
    "Unable to query nvcc version from ${CUDA_BIN_DIR}/nvcc."
  [[ "${ACTUAL_CUDA_VERSION}" == "${CUDA_VERSION}" ]] || fail \
    "CUDA toolkit mismatch: expected ${CUDA_VERSION}, found ${ACTUAL_CUDA_VERSION:-unknown}."
}

verify_driver_pin() {
  ACTUAL_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | tr -d ' ')" || fail \
    "Unable to query NVIDIA driver version via nvidia-smi."
  [[ -n "${ACTUAL_DRIVER_VERSION}" ]] || fail "nvidia-smi did not report a driver version."
  version_at_least "${ACTUAL_DRIVER_VERSION}" "${DRIVER_MINIMUM}" || fail \
    "Driver version ${ACTUAL_DRIVER_VERSION} does not satisfy minimum ${DRIVER_MINIMUM}."
}

verify_git_pin() {
  local repo_dir="$1"
  local expected_ref="$2"
  local label="$3"
  local actual_ref

  actual_ref="$(git -C "${repo_dir}" rev-parse HEAD)" || fail \
    "Unable to query ${label} git ref in ${repo_dir}."
  [[ "${actual_ref}" == "${expected_ref}" ]] || fail \
    "${label} mismatch: expected ${expected_ref}, found ${actual_ref}."

  printf '%s' "${actual_ref}"
}

verify_nvbandwidth_pin() {
  ACTUAL_NVBANDWIDTH_HEAD="$(git -C "${NVBANDWIDTH_REPO_DIR}" rev-parse HEAD)" || fail \
    "Unable to query nvbandwidth git ref in ${NVBANDWIDTH_REPO_DIR}."
  ACTUAL_NVBANDWIDTH_TAG="$(git -C "${NVBANDWIDTH_REPO_DIR}" tag --points-at HEAD | grep -Fx "${NVBANDWIDTH_TAG}" || true)"
  [[ -n "${ACTUAL_NVBANDWIDTH_TAG}" ]] || fail \
    "nvbandwidth checkout mismatch: expected tag ${NVBANDWIDTH_TAG} on HEAD ${ACTUAL_NVBANDWIDTH_HEAD}."
}

verify_nccl_pin() {
  ACTUAL_LIBNCCL2_VERSION="$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || true)"
  ACTUAL_LIBNCCL_DEV_VERSION="$(dpkg-query -W -f='${Version}' libnccl-dev 2>/dev/null || true)"
  [[ "${ACTUAL_LIBNCCL2_VERSION}" == "${NCCL_DEV_VERSION}" ]] || fail \
    "libnccl2 version mismatch: expected ${NCCL_DEV_VERSION}, found ${ACTUAL_LIBNCCL2_VERSION:-missing}."
  [[ "${ACTUAL_LIBNCCL_DEV_VERSION}" == "${NCCL_DEV_VERSION}" ]] || fail \
    "libnccl-dev version mismatch: expected ${NCCL_DEV_VERSION}, found ${ACTUAL_LIBNCCL_DEV_VERSION:-missing}."
}

verify_toolchain() {
  ACTUAL_MAMF_PATH="${MAMF_FINDER_PATH}"
  ACTUAL_MAMF_COMMIT="$(verify_git_pin "${MAMF_FINDER_REPO_DIR}" "${MAMF_FINDER_COMMIT}" "mamf-finder repo")"
  [[ "${ACTUAL_MAMF_PATH}" == "${MAMF_FINDER_REPO_DIR}/${MAMF_FINDER_RELPATH}" ]] || fail \
    "mamf-finder path mismatch: expected ${MAMF_FINDER_REPO_DIR}/${MAMF_FINDER_RELPATH}, found ${ACTUAL_MAMF_PATH}."
  ACTUAL_NCCL_TESTS_COMMIT="$(verify_git_pin "${NCCL_TESTS_REPO_DIR}" "${NCCL_TESTS_COMMIT}" "nccl-tests repo")"
  verify_python_pin
  verify_ornn_bench_pin
  verify_torch_pin
  verify_cuda_pin
  verify_driver_pin
  verify_nvbandwidth_pin
  verify_nccl_pin
  write_toolchain_manifest
}

write_toolchain_manifest() {
  {
    write_env_entry "ORNN_BENCH_BIN" "${ORNN_BENCH_BIN}"
    write_env_entry "ORNN_BENCH_EXPECTED_VERSION" "${ORNN_BENCH_VERSION}"
    write_env_entry "ORNN_BENCH_ACTUAL_VERSION" "${ACTUAL_ORNN_BENCH_VERSION}"
    write_env_entry "PYTHON_EXPECTED_VERSION" "${PYTHON_VERSION}"
    write_env_entry "PYTHON_ACTUAL_VERSION" "${ACTUAL_PYTHON_VERSION}"
    write_env_entry "TORCH_EXPECTED_VERSION" "${TORCH_VERSION}+${TORCH_CUDA_WHEEL}"
    write_env_entry "TORCH_ACTUAL_VERSION" "${ACTUAL_TORCH_VERSION}"
    write_env_entry "TORCH_ACTUAL_CUDA" "${ACTUAL_TORCH_CUDA}"
    write_env_entry "CUDA_EXPECTED_VERSION" "${CUDA_VERSION}"
    write_env_entry "CUDA_ACTUAL_VERSION" "${ACTUAL_CUDA_VERSION}"
    write_env_entry "DRIVER_MINIMUM" "${DRIVER_MINIMUM}"
    write_env_entry "DRIVER_ACTUAL_VERSION" "${ACTUAL_DRIVER_VERSION}"
    write_env_entry "MAMF_FINDER_EXPECTED_COMMIT" "${MAMF_FINDER_COMMIT}"
    write_env_entry "MAMF_FINDER_ACTUAL_COMMIT" "${ACTUAL_MAMF_COMMIT}"
    write_env_entry "MAMF_FINDER_EXPECTED_PATH" "${MAMF_FINDER_REPO_DIR}/${MAMF_FINDER_RELPATH}"
    write_env_entry "MAMF_FINDER_ACTUAL_PATH" "${ACTUAL_MAMF_PATH}"
    write_env_entry "NVBANDWIDTH_EXPECTED_TAG" "${NVBANDWIDTH_TAG}"
    write_env_entry "NVBANDWIDTH_ACTUAL_TAG" "${ACTUAL_NVBANDWIDTH_TAG}"
    write_env_entry "NVBANDWIDTH_ACTUAL_HEAD" "${ACTUAL_NVBANDWIDTH_HEAD}"
    write_env_entry "NCCL_TESTS_EXPECTED_COMMIT" "${NCCL_TESTS_COMMIT}"
    write_env_entry "NCCL_TESTS_ACTUAL_COMMIT" "${ACTUAL_NCCL_TESTS_COMMIT}"
    write_env_entry "NCCL_EXPECTED_VERSION" "${NCCL_DEV_VERSION}"
    write_env_entry "NCCL_ACTUAL_LIBNCCL2_VERSION" "${ACTUAL_LIBNCCL2_VERSION}"
    write_env_entry "NCCL_ACTUAL_LIBNCCL_DEV_VERSION" "${ACTUAL_LIBNCCL_DEV_VERSION}"
  } > "${TOOLCHAIN_MANIFEST_PATH}"
}

record_gpu_identity() {
  local gpu_index="$1"
  local gpu_name="$2"
  local gpu_memory="$3"

  case "${gpu_index}" in
    1)
      GPU_NAME_1="${gpu_name}"
      GPU_MEMORY_1="${gpu_memory}"
      ;;
    2)
      GPU_NAME_2="${gpu_name}"
      GPU_MEMORY_2="${gpu_memory}"
      ;;
    3)
      GPU_NAME_3="${gpu_name}"
      GPU_MEMORY_3="${gpu_memory}"
      ;;
    4)
      GPU_NAME_4="${gpu_name}"
      GPU_MEMORY_4="${gpu_memory}"
      ;;
    5)
      GPU_NAME_5="${gpu_name}"
      GPU_MEMORY_5="${gpu_memory}"
      ;;
    6)
      GPU_NAME_6="${gpu_name}"
      GPU_MEMORY_6="${gpu_memory}"
      ;;
    7)
      GPU_NAME_7="${gpu_name}"
      GPU_MEMORY_7="${gpu_memory}"
      ;;
    8)
      GPU_NAME_8="${gpu_name}"
      GPU_MEMORY_8="${gpu_memory}"
      ;;
  esac
}

print_topology() {
  if [[ -z "${TOPOLOGY_OUTPUT}" ]]; then
    TOPOLOGY_OUTPUT="$(nvidia-smi topo -m 2>&1 || true)"
    if [[ -n "${TOPOLOGY_OUTPUT}" && -n "${TOPOLOGY_FILE_PATH}" ]]; then
      printf '%s\n' "${TOPOLOGY_OUTPUT}" > "${TOPOLOGY_FILE_PATH}"
    fi
  fi
  if [[ -n "${TOPOLOGY_OUTPUT}" ]]; then
    printf '%s\n' "nvidia-smi topo -m output:" >&2
    printf '%s\n' "${TOPOLOGY_OUTPUT}" >&2
  fi
}

fail_sxm5_gate() {
  print_topology
  fail "$*"
}

capture_gpu_inventory() {
  local gpu_output
  local raw_line
  local gpu_name
  local gpu_memory
  local gpu_driver

  gpu_output="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)" || fail \
    "Unable to query GPU inventory via nvidia-smi."

  GPU_COUNT=0
  printf 'gpu_index,name,memory_total_mib,driver_version\n' > "${GPU_INVENTORY_PATH}"
  while IFS= read -r raw_line; do
    [[ -n "$(trim "${raw_line}")" ]] || continue
    gpu_name="$(trim "$(printf '%s' "${raw_line}" | cut -d',' -f1)")"
    gpu_memory="$(trim "$(printf '%s' "${raw_line}" | cut -d',' -f2)")"
    gpu_driver="$(trim "$(printf '%s' "${raw_line}" | cut -d',' -f3)")"

    GPU_COUNT=$((GPU_COUNT + 1))
    printf '%s,%s,%s,%s\n' "${GPU_COUNT}" "${gpu_name}" "${gpu_memory}" "${gpu_driver}" >> "${GPU_INVENTORY_PATH}"
    record_gpu_identity "${GPU_COUNT}" "${gpu_name}" "${gpu_memory}"

    [[ -n "${gpu_memory}" ]] || fail_sxm5_gate "GPU ${GPU_COUNT} did not report total memory."
    if [[ "${gpu_memory}" -lt "${MEMORY_FLOOR_MIB}" ]]; then
      fail_sxm5_gate \
        "GPU ${GPU_COUNT} reported ${gpu_memory} MiB, below the ${MEMORY_FLOOR_MIB} MiB SXM5 floor."
    fi
  done <<< "${gpu_output}"

  [[ "${GPU_COUNT}" -eq "${EXPECTED_GPU_COUNT}" ]] || fail_sxm5_gate \
    "Expected ${EXPECTED_GPU_COUNT} GPUs, found ${GPU_COUNT}."
}

capture_topology() {
  local raw_line
  local line
  local row_gpu
  local token
  local token_count
  local row_has_nv

  TOPOLOGY_OUTPUT="$(nvidia-smi topo -m)" || fail_sxm5_gate "Unable to query nvidia-smi topo -m."
  printf '%s\n' "${TOPOLOGY_OUTPUT}" > "${TOPOLOGY_FILE_PATH}"

  TOPOLOGY_GPU_ROW_COUNT=0
  TOPOLOGY_ROWS_WITH_NV=0
  while IFS= read -r raw_line; do
    line="$(trim "${raw_line}")"
    set -- ${line}
    [[ "$#" -ge 2 ]] || continue
    row_gpu="$1"
    [[ "${row_gpu}" =~ ^GPU[0-9]+$ ]] || continue
    [[ "$2" =~ ^GPU[0-9]+$ ]] && continue

    TOPOLOGY_GPU_ROW_COUNT=$((TOPOLOGY_GPU_ROW_COUNT + 1))
    shift
    token_count=0
    row_has_nv=0
    while [[ "$#" -gt 0 && "${token_count}" -lt "${EXPECTED_GPU_COUNT}" ]]; do
      token="$1"
      shift
      token_count=$((token_count + 1))
      if [[ "${token}" == NV* ]]; then
        row_has_nv=1
      fi
    done

    if [[ "${row_has_nv}" -eq 1 ]]; then
      TOPOLOGY_ROWS_WITH_NV=$((TOPOLOGY_ROWS_WITH_NV + 1))
    fi
  done <<< "${TOPOLOGY_OUTPUT}"

  [[ "${TOPOLOGY_GPU_ROW_COUNT}" -eq "${EXPECTED_GPU_COUNT}" ]] || fail_sxm5_gate \
    "Expected ${EXPECTED_GPU_COUNT} GPU rows in nvidia-smi topo -m, found ${TOPOLOGY_GPU_ROW_COUNT}."
  [[ "${TOPOLOGY_ROWS_WITH_NV}" -eq "${EXPECTED_GPU_COUNT}" ]] || fail_sxm5_gate \
    "Expected every GPU row to expose NV* connectivity; topology appears PCIe-only or incomplete."
}

write_sxm5_manifest() {
  {
    write_env_entry "EXPECTED_GPU_COUNT" "${EXPECTED_GPU_COUNT}"
    write_env_entry "ACTUAL_GPU_COUNT" "${GPU_COUNT}"
    write_env_entry "MEMORY_FLOOR_MIB" "${MEMORY_FLOOR_MIB}"
    write_env_entry "GPU_1_NAME" "${GPU_NAME_1}"
    write_env_entry "GPU_1_MEMORY_MIB" "${GPU_MEMORY_1}"
    write_env_entry "GPU_2_NAME" "${GPU_NAME_2}"
    write_env_entry "GPU_2_MEMORY_MIB" "${GPU_MEMORY_2}"
    write_env_entry "GPU_3_NAME" "${GPU_NAME_3}"
    write_env_entry "GPU_3_MEMORY_MIB" "${GPU_MEMORY_3}"
    write_env_entry "GPU_4_NAME" "${GPU_NAME_4}"
    write_env_entry "GPU_4_MEMORY_MIB" "${GPU_MEMORY_4}"
    write_env_entry "GPU_5_NAME" "${GPU_NAME_5}"
    write_env_entry "GPU_5_MEMORY_MIB" "${GPU_MEMORY_5}"
    write_env_entry "GPU_6_NAME" "${GPU_NAME_6}"
    write_env_entry "GPU_6_MEMORY_MIB" "${GPU_MEMORY_6}"
    write_env_entry "GPU_7_NAME" "${GPU_NAME_7}"
    write_env_entry "GPU_7_MEMORY_MIB" "${GPU_MEMORY_7}"
    write_env_entry "GPU_8_NAME" "${GPU_NAME_8}"
    write_env_entry "GPU_8_MEMORY_MIB" "${GPU_MEMORY_8}"
    write_env_entry "TOPOLOGY_GPU_ROW_COUNT" "${TOPOLOGY_GPU_ROW_COUNT}"
    write_env_entry "TOPOLOGY_ROWS_WITH_NV" "${TOPOLOGY_ROWS_WITH_NV}"
    write_env_entry "TOPOLOGY_FILE" "${TOPOLOGY_FILE_PATH}"
  } > "${SXM5_MANIFEST_PATH}"
}

enforce_sxm5_gate() {
  capture_gpu_inventory
  capture_topology
  write_sxm5_manifest
}

recover_default_report() {
  local work_dir="$1"
  local requested_path="$2"
  local candidates=()

  shopt -s nullglob
  candidates=("${work_dir}"/ornn_report_*.json)
  shopt -u nullglob

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    fail "Expected report at ${requested_path}, but no fallback ornn_report_*.json was produced in ${work_dir}."
  fi
  if [[ "${#candidates[@]}" -gt 1 ]]; then
    fail \
      "Expected report at ${requested_path}, but multiple fallback ornn_report_*.json files were produced in ${work_dir}."
  fi

  mv "${candidates[0]}" "${requested_path}"
  warn \
    "Expected report at ${requested_path}, but ornn-bench wrote ${candidates[0]}. Recovered the default report into place."
}

verify_report_payload() {
  local report_path="$1"

  "${ORNN_BENCH_PYTHON}" - "${report_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
payload = json.loads(report_path.read_text())
scores = payload.get("scores")
manifest = payload.get("manifest")

if not isinstance(scores, dict) or not scores:
    raise SystemExit(f"report missing populated scores: {report_path}")
for key in ("ornn_i", "ornn_t", "status", "components"):
    if key not in scores:
        raise SystemExit(f"report scores missing '{key}': {report_path}")
if not isinstance(manifest, dict) or not manifest:
    raise SystemExit(f"report missing populated manifest: {report_path}")
PY
}

store_run_artifacts() {
  local run_number="$1"
  local report_path="$2"
  local log_path="$3"

  case "${run_number}" in
    1)
      RUN_1_REPORT="${report_path}"
      RUN_1_LOG="${log_path}"
      ;;
    2)
      RUN_2_REPORT="${report_path}"
      RUN_2_LOG="${log_path}"
      ;;
    3)
      RUN_3_REPORT="${report_path}"
      RUN_3_LOG="${log_path}"
      ;;
  esac
}

write_run_manifest() {
  local run_number="$1"
  local requested_path="$2"
  local log_path="$3"
  local work_dir="$4"
  local recovered_from_default="$5"
  local exit_code="$6"
  local manifest_path
  local report_status

  manifest_path="${MANIFESTS_DIR}/run-0${run_number}.env"
  if [[ -f "${requested_path}" ]]; then
    report_status="present"
  else
    report_status="missing"
  fi

  {
    write_env_entry "RUN_NUMBER" "${run_number}"
    write_env_entry "STATUS" "$(if [[ "${exit_code}" -eq 0 ]]; then printf 'completed'; else printf 'failed'; fi)"
    write_env_entry "EXIT_CODE" "${exit_code}"
    write_env_entry "REPORT_PATH" "${requested_path}"
    write_env_entry "REPORT_STATUS" "${report_status}"
    write_env_entry "LOG_PATH" "${log_path}"
    write_env_entry "WORK_DIR" "${work_dir}"
    write_env_entry "RECOVERED_FROM_DEFAULT" "${recovered_from_default}"
    write_env_entry "UPLOAD_ENABLED" "${UPLOAD_ENABLED}"
    write_env_entry "COMMAND" "${ORNN_BENCH_BIN} run -o ${requested_path}$(if [[ "${UPLOAD_ENABLED}" -eq 1 ]]; then printf ' --upload'; fi)"
  } > "${manifest_path}"
}

run_single_capture() {
  local run_number="$1"
  local run_label
  local work_dir
  local report_path
  local log_path
  local exit_code
  local recovered_from_default="0"
  local cmd

  run_label="run-0${run_number}"
  work_dir="${WORK_DIR}/${run_label}"
  report_path="${REPORTS_DIR}/${run_label}.json"
  log_path="${LOGS_DIR}/${run_label}.log"
  mkdir -p "${work_dir}"

  cmd="${ORNN_BENCH_BIN} run -o ${report_path}"
  if [[ "${UPLOAD_ENABLED}" -eq 1 ]]; then
    cmd="${cmd} --upload"
  fi

  log "Starting ${run_label}"
  set +e
  (
    cd "${work_dir}"
    if [[ "${UPLOAD_ENABLED}" -eq 1 ]]; then
      "${ORNN_BENCH_BIN}" run -o "${report_path}" --upload
    else
      "${ORNN_BENCH_BIN}" run -o "${report_path}"
    fi
  ) 2>&1 | tee "${log_path}"
  exit_code=$?
  set -e

  if [[ ! -f "${report_path}" ]]; then
    recover_default_report "${work_dir}" "${report_path}"
    recovered_from_default="1"
  fi

  if [[ -f "${report_path}" ]]; then
    verify_report_payload "${report_path}" || fail \
      "Report verification failed for ${run_label} at ${report_path}."
  fi

  write_run_manifest \
    "${run_number}" \
    "${report_path}" \
    "${log_path}" \
    "${work_dir}" \
    "${recovered_from_default}" \
    "${exit_code}"
  store_run_artifacts "${run_number}" "${report_path}" "${log_path}"

  if [[ "${exit_code}" -ne 0 ]]; then
    fail "ornn-bench run failed for ${run_label} with exit code ${exit_code}."
  fi

  COMPLETED_RUNS="${run_number}"
  write_campaign_manifest

  if [[ "${run_number}" -lt "${RUN_COUNT}" ]]; then
    log "Cooling down for ${COOLDOWN_SECONDS} seconds"
    sleep "${COOLDOWN_SECONDS}"
  fi
}

main() {
  [[ "$#" -eq 3 ]] || usage
  [[ "${COOLDOWN_SECONDS}" =~ ^[0-9]+$ ]] || fail \
    "ORNN_BENCH_COOLDOWN_SECONDS must be an integer number of seconds."

  PROVIDER="$1"
  REGION="$2"
  INSTANCE_TYPE="$3"

  initialize_campaign_layout
  CAMPAIGN_STATUS="verifying"
  write_campaign_manifest

  source_locked_env
  prepare_runtime_paths
  resolve_upload_mode
  verify_toolchain
  enforce_sxm5_gate

  CAMPAIGN_STATUS="running"
  write_campaign_manifest
  run_single_capture 1
  run_single_capture 2
  run_single_capture 3

  CAMPAIGN_STATUS="completed"
  write_campaign_manifest
  log "Benchmark campaign complete"
  log "Reports stored in ${REPORTS_DIR}"
  log "Logs stored in ${LOGS_DIR}"
  log "Manifests stored in ${MANIFESTS_DIR}"
}

main "$@"
