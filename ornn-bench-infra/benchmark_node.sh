#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Placeholder scaffold for bench-infra-benchmark-node-runner.
# Future implementation must verify "${SCRIPT_DIR}/toolchain.env", enforce the
# SXM5 host gate, and run the benchmark cadence with explicit output paths.

printf '%s\n' \
  "benchmark_node.sh is a placeholder scaffold; implementation lands in feature bench-infra-benchmark-node-runner." \
  >&2
printf '%s\n' "Locked pins are defined in ${SCRIPT_DIR}/toolchain.env." >&2
exit 1
