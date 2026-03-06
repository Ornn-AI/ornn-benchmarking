#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Placeholder scaffold for bench-infra-provision-script.
# Future implementation must source "${SCRIPT_DIR}/toolchain.env" and write
# provider-specific discoveries to /opt/ornn-bench-tools/runtime.env without
# modifying the checked-in toolchain lock.

printf '%s\n' \
  "provision.sh is a placeholder scaffold; implementation lands in feature bench-infra-provision-script." \
  >&2
printf '%s\n' "Locked pins are defined in ${SCRIPT_DIR}/toolchain.env." >&2
exit 1
