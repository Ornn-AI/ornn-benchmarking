#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN_ENV="${SCRIPT_DIR}/toolchain.env"

# shellcheck disable=SC1091
source "${TOOLCHAIN_ENV}"

readonly TOOLCHAIN_ENV
readonly TOOLS_ROOT="/opt/ornn-bench-tools"
readonly RUNTIME_ENV="${TOOLS_ROOT}/runtime.env"
readonly SRC_ROOT="${TOOLS_ROOT}/src"
readonly BIN_ROOT="${TOOLS_ROOT}/bin"
readonly VENV_DIR="${TOOLS_ROOT}/venv"
readonly NVBANDWIDTH_REPO="https://github.com/NVIDIA/nvbandwidth.git"
readonly NCCL_TESTS_REPO="https://github.com/NVIDIA/nccl-tests.git"
readonly CUDA_APT_SUFFIX="${CUDA_VERSION//./-}"

APT_UPDATED=0
PYTHON_BIN=""
VENV_PYTHON=""
CUDA_HOME=""
CUDA_BIN_DIR=""
CUDA_LIB_DIR=""
NCCL_HOME=""
NCCL_LIB_DIR=""
MAMF_REPO_DIR="${SRC_ROOT}/ml-engineering"
MAMF_FINDER_PATH=""
NVBANDWIDTH_REPO_DIR="${SRC_ROOT}/nvbandwidth"
NVBANDWIDTH_BUILD_DIR="${NVBANDWIDTH_REPO_DIR}/build"
NVBANDWIDTH_PATH=""
NCCL_TESTS_REPO_DIR="${SRC_ROOT}/nccl-tests"
NCCL_TESTS_BIN_DIR="${NCCL_TESTS_REPO_DIR}/build"

log() {
  printf '==> %s\n' "$*" >&2
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

apt_update_once() {
  if [[ "${APT_UPDATED}" -eq 0 ]]; then
    log "Refreshing apt package metadata"
    DEBIAN_FRONTEND=noninteractive apt-get update -y
    APT_UPDATED=1
  fi
}

install_apt_packages() {
  apt_update_once
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
}

install_pinned_apt_packages() {
  apt_update_once
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    --allow-downgrades \
    --allow-change-held-packages \
    "$@"
}

require_linux() {
  [[ "$(uname -s)" == "Linux" ]] || fail "provision.sh only supports Linux provider GPU images."
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || fail "Run provision.sh as root (for apt installs and /opt writes)."
}

require_apt() {
  command -v apt-get >/dev/null 2>&1 || fail \
    "provision.sh currently supports apt-based Linux images because pinned NCCL packages are distributed via apt."
}

package_version() {
  dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true
}

ensure_apt_package_version_available() {
  local package_name="$1"
  local version="$2"
  local package_label="$3"

  apt_update_once
  if ! apt-cache madison "${package_name}" | awk '{print $3}' | grep -Fxq "${version}"; then
    fail \
      "Pinned ${package_label} package ${package_name}=${version} is unavailable. Ensure the NVIDIA CUDA apt repository for CUDA ${CUDA_VERSION} is configured on this image, then re-run provision.sh."
  fi
}

ensure_base_layout() {
  install -d -m 0755 "${TOOLS_ROOT}" "${SRC_ROOT}" "${BIN_ROOT}"
}

ensure_build_dependencies() {
  log "Installing build dependencies"
  install_apt_packages \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    libboost-program-options-dev
}

ensure_python_toolchain() {
  local python_pkg="python${PYTHON_VERSION}"
  local venv_pkg="${python_pkg}-venv"

  if ! command -v "${python_pkg}" >/dev/null 2>&1; then
    log "Installing pinned Python ${PYTHON_VERSION}"
    install_apt_packages "${python_pkg}" "${venv_pkg}"
  fi

  PYTHON_BIN="$(command -v "${python_pkg}" || true)"
  [[ -n "${PYTHON_BIN}" ]] || fail "Pinned Python ${PYTHON_VERSION} is not available on this image."

  local reported_python
  reported_python="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${reported_python}" == "${PYTHON_VERSION}" ]] || fail \
    "Expected Python ${PYTHON_VERSION}, found ${reported_python} at ${PYTHON_BIN}."

  ensure_base_layout

  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    local existing_venv_python
    existing_venv_python="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${existing_venv_python}" != "${PYTHON_VERSION}" ]]; then
      log "Recreating virtualenv to match Python ${PYTHON_VERSION}"
      rm -rf "${VENV_DIR}"
    fi
  fi

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  VENV_PYTHON="${VENV_DIR}/bin/python"
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
  ln -sf "${VENV_DIR}/bin/ornn-bench" "${BIN_ROOT}/ornn-bench"
}

detect_cuda_home() {
  if [[ -d "/usr/local/cuda-${CUDA_VERSION}" ]]; then
    CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
  elif [[ -d "/usr/local/cuda" ]]; then
    CUDA_HOME="/usr/local/cuda"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)"
  else
    fail "Unable to locate a CUDA ${CUDA_VERSION} toolkit (nvcc is missing)."
  fi

  CUDA_BIN_DIR="${CUDA_HOME}/bin"
  CUDA_LIB_DIR="${CUDA_HOME}/lib64"
  [[ -x "${CUDA_BIN_DIR}/nvcc" ]] || fail "Expected nvcc at ${CUDA_BIN_DIR}/nvcc."
}

ensure_cuda_toolkit() {
  local installed_cuda=""
  if command -v nvcc >/dev/null 2>&1; then
    installed_cuda="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
  fi

  if [[ "${installed_cuda}" != "${CUDA_VERSION}" ]]; then
    log "Installing CUDA toolkit ${CUDA_VERSION}"
    install_apt_packages "cuda-toolkit-${CUDA_APT_SUFFIX}"
  fi

  detect_cuda_home
  local verified_cuda
  verified_cuda="$("${CUDA_BIN_DIR}/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
  [[ "${verified_cuda}" == "${CUDA_VERSION}" ]] || fail \
    "Expected CUDA toolkit ${CUDA_VERSION}, found ${verified_cuda:-unknown}."
}

ensure_nccl_packages() {
  ensure_apt_package_version_available "libnccl2" "${NCCL_DEV_VERSION}" "NCCL runtime"
  ensure_apt_package_version_available "libnccl-dev" "${NCCL_DEV_VERSION}" "NCCL development"

  log "Installing pinned NCCL packages"
  install_pinned_apt_packages \
    "libnccl2=${NCCL_DEV_VERSION}" \
    "libnccl-dev=${NCCL_DEV_VERSION}"

  [[ "$(package_version libnccl2)" == "${NCCL_DEV_VERSION}" ]] || fail \
    "Installed libnccl2 version does not match ${NCCL_DEV_VERSION}."
  [[ "$(package_version libnccl-dev)" == "${NCCL_DEV_VERSION}" ]] || fail \
    "Installed libnccl-dev version does not match ${NCCL_DEV_VERSION}."

  NCCL_HOME="/usr"
  NCCL_LIB_DIR="$(dpkg -L libnccl-dev | awk '/\/libnccl\.so$/ {print; exit}')"
  [[ -n "${NCCL_LIB_DIR}" ]] || fail "Unable to locate libnccl.so after installing libnccl-dev."
  NCCL_LIB_DIR="$(dirname "${NCCL_LIB_DIR}")"
}

sync_git_checkout() {
  local repo_url="$1"
  local dest_dir="$2"
  local ref="$3"

  if [[ -e "${dest_dir}" && ! -d "${dest_dir}/.git" ]]; then
    rm -rf "${dest_dir}"
  fi

  if [[ ! -d "${dest_dir}/.git" ]]; then
    log "Cloning ${repo_url}"
    git clone "${repo_url}" "${dest_dir}"
  fi

  git -C "${dest_dir}" remote set-url origin "${repo_url}"
  git -C "${dest_dir}" fetch --tags --force origin
  git -C "${dest_dir}" checkout --detach "${ref}"
  git -C "${dest_dir}" reset --hard "${ref}"
  git -C "${dest_dir}" clean -fdx
}

verify_git_ref() {
  local repo_dir="$1"
  local expected_ref="$2"
  local label="$3"
  local actual_ref

  actual_ref="$(git -C "${repo_dir}" rev-parse HEAD)"
  [[ "${actual_ref}" == "${expected_ref}" ]] || fail \
    "${label} expected ${expected_ref}, found ${actual_ref}."
}

install_ornn_bench() {
  log "Installing ornn-bench ${ORNN_BENCH_VERSION}"
  "${VENV_PYTHON}" -m pip install --upgrade --force-reinstall --no-cache-dir \
    "ornn-bench==${ORNN_BENCH_VERSION}"
}

verify_ornn_bench() {
  local reported_version
  reported_version="$(${VENV_DIR}/bin/ornn-bench --version | awk '{print $2}')"
  [[ "${reported_version}" == "${ORNN_BENCH_VERSION}" ]] || fail \
    "ornn-bench version mismatch: expected ${ORNN_BENCH_VERSION}, found ${reported_version}."
  ln -sf "${VENV_DIR}/bin/ornn-bench" "${BIN_ROOT}/ornn-bench"
}

install_torch() {
  local torch_index_url="https://download.pytorch.org/whl/${TORCH_CUDA_WHEEL}"
  log "Installing torch ${TORCH_VERSION}+${TORCH_CUDA_WHEEL}"
  "${VENV_PYTHON}" -m pip install --upgrade --force-reinstall --no-cache-dir \
    --index-url "${torch_index_url}" \
    "torch==${TORCH_VERSION}"
}

verify_torch() {
  "${VENV_PYTHON}" - "${TORCH_VERSION}" "${TORCH_CUDA_WHEEL}" "${CUDA_VERSION}" <<'PY'
import sys

import torch

expected_version, expected_wheel, expected_cuda = sys.argv[1:4]
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
}

install_mamf_finder() {
  log "Installing mamf-finder at pinned commit ${MAMF_FINDER_COMMIT}"
  sync_git_checkout "${MAMF_FINDER_REPO}" "${MAMF_REPO_DIR}" "${MAMF_FINDER_COMMIT}"
  verify_git_ref "${MAMF_REPO_DIR}" "${MAMF_FINDER_COMMIT}" "mamf-finder repo"

  MAMF_FINDER_PATH="${MAMF_REPO_DIR}/${MAMF_FINDER_RELPATH}"
  [[ -f "${MAMF_FINDER_PATH}" ]] || fail \
    "mamf-finder script not found at ${MAMF_FINDER_PATH}."
  ln -sf "${MAMF_FINDER_PATH}" "${BIN_ROOT}/mamf-finder.py"
}

install_nvbandwidth() {
  log "Installing nvbandwidth tag ${NVBANDWIDTH_TAG}"
  sync_git_checkout "${NVBANDWIDTH_REPO}" "${NVBANDWIDTH_REPO_DIR}" "${NVBANDWIDTH_TAG}"

  if ! git -C "${NVBANDWIDTH_REPO_DIR}" tag --points-at HEAD | grep -Fxq "${NVBANDWIDTH_TAG}"; then
    fail "nvbandwidth checkout is not pinned to tag ${NVBANDWIDTH_TAG}."
  fi

  env PATH="${CUDA_BIN_DIR}:${PATH}" \
    CUDACXX="${CUDA_BIN_DIR}/nvcc" \
    cmake -S "${NVBANDWIDTH_REPO_DIR}" -B "${NVBANDWIDTH_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  env PATH="${CUDA_BIN_DIR}:${PATH}" \
    cmake --build "${NVBANDWIDTH_BUILD_DIR}" --parallel "$(nproc)"

  NVBANDWIDTH_PATH="${NVBANDWIDTH_BUILD_DIR}/nvbandwidth"
  [[ -x "${NVBANDWIDTH_PATH}" ]] || fail \
    "nvbandwidth binary not found at ${NVBANDWIDTH_PATH}."
  ln -sf "${NVBANDWIDTH_PATH}" "${BIN_ROOT}/nvbandwidth"
}

install_nccl_tests() {
  log "Installing nccl-tests at pinned commit ${NCCL_TESTS_COMMIT}"
  sync_git_checkout "${NCCL_TESTS_REPO}" "${NCCL_TESTS_REPO_DIR}" "${NCCL_TESTS_COMMIT}"
  verify_git_ref "${NCCL_TESTS_REPO_DIR}" "${NCCL_TESTS_COMMIT}" "nccl-tests repo"

  env PATH="${CUDA_BIN_DIR}:${PATH}" \
    make -C "${NCCL_TESTS_REPO_DIR}" -j"$(nproc)" \
      CUDA_HOME="${CUDA_HOME}" \
      NCCL_HOME="${NCCL_HOME}"

  for binary_name in \
    all_reduce_perf \
    reduce_scatter_perf \
    all_gather_perf \
    broadcast_perf \
    sendrecv_perf
  do
    [[ -x "${NCCL_TESTS_BIN_DIR}/${binary_name}" ]] || fail \
      "Expected nccl-tests binary ${NCCL_TESTS_BIN_DIR}/${binary_name}."
    ln -sf "${NCCL_TESTS_BIN_DIR}/${binary_name}" "${BIN_ROOT}/${binary_name}"
  done
}

write_runtime_env() {
  local temp_env
  temp_env="$(mktemp "${TOOLS_ROOT}/runtime.env.XXXXXX")"

  {
    printf '# Generated by %s\n' "${BASH_SOURCE[0]}"
    printf '# Source this file after toolchain.env to discover provisioned paths.\n'
    printf 'ORNN_BENCH_TOOLS_ROOT="%s"\n' "${TOOLS_ROOT}"
    printf 'ORNN_BENCH_VENV="%s"\n' "${VENV_DIR}"
    printf 'ORNN_BENCH_PYTHON="%s"\n' "${VENV_PYTHON}"
    printf 'TOOLS_BIN_DIR="%s"\n' "${BIN_ROOT}"
    printf 'ORNN_BENCH_BIN="%s"\n' "${BIN_ROOT}/ornn-bench"
    printf 'MAMF_FINDER_REPO_DIR="%s"\n' "${MAMF_REPO_DIR}"
    printf 'MAMF_FINDER_PATH="%s"\n' "${MAMF_FINDER_PATH}"
    printf 'NVBANDWIDTH_REPO_DIR="%s"\n' "${NVBANDWIDTH_REPO_DIR}"
    printf 'NVBANDWIDTH_BUILD_DIR="%s"\n' "${NVBANDWIDTH_BUILD_DIR}"
    printf 'NVBANDWIDTH_PATH="%s"\n' "${BIN_ROOT}/nvbandwidth"
    printf 'NCCL_TESTS_REPO_DIR="%s"\n' "${NCCL_TESTS_REPO_DIR}"
    printf 'NCCL_TESTS_BIN_DIR="%s"\n' "${NCCL_TESTS_BIN_DIR}"
    printf 'ALL_REDUCE_PERF_PATH="%s"\n' "${BIN_ROOT}/all_reduce_perf"
    printf 'REDUCE_SCATTER_PERF_PATH="%s"\n' "${BIN_ROOT}/reduce_scatter_perf"
    printf 'ALL_GATHER_PERF_PATH="%s"\n' "${BIN_ROOT}/all_gather_perf"
    printf 'BROADCAST_PERF_PATH="%s"\n' "${BIN_ROOT}/broadcast_perf"
    printf 'SENDRECV_PERF_PATH="%s"\n' "${BIN_ROOT}/sendrecv_perf"
    printf 'CUDA_HOME="%s"\n' "${CUDA_HOME}"
    printf 'CUDA_BIN_DIR="%s"\n' "${CUDA_BIN_DIR}"
    printf 'CUDA_LIB_DIR="%s"\n' "${CUDA_LIB_DIR}"
    printf 'NCCL_HOME="%s"\n' "${NCCL_HOME}"
    printf 'NCCL_LIB_DIR="%s"\n' "${NCCL_LIB_DIR}"
  } > "${temp_env}"

  mv "${temp_env}" "${RUNTIME_ENV}"
  chmod 0644 "${RUNTIME_ENV}"
}

verify_runtime_env() {
  (
    set -euo pipefail
    # shellcheck disable=SC1090
    source "${RUNTIME_ENV}"
    [[ -x "${ORNN_BENCH_BIN}" ]]
    [[ -f "${MAMF_FINDER_PATH}" ]]
    [[ -x "${NVBANDWIDTH_PATH}" ]]
    [[ -x "${ALL_REDUCE_PERF_PATH}" ]]
  )
}

main() {
  require_linux
  require_root
  require_apt
  ensure_build_dependencies
  ensure_python_toolchain
  ensure_cuda_toolkit
  ensure_nccl_packages
  install_ornn_bench
  verify_ornn_bench
  install_torch
  verify_torch
  install_mamf_finder
  install_nvbandwidth
  install_nccl_tests
  write_runtime_env
  verify_runtime_env

  log "Provisioning complete"
  log "Toolchain lock: ${TOOLCHAIN_ENV}"
  log "Runtime paths written to: ${RUNTIME_ENV}"
}

main "$@"
