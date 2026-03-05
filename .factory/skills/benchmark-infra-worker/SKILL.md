---
name: benchmark-infra-worker
description: Implements benchmark campaign shell tooling (toolchain.env, provision.sh, benchmark_node.sh) with strict reproducibility guarantees.
---

# Benchmark Infra Worker

NOTE: Startup/cleanup are handled by worker-base. This skill defines the work procedure.

## When to Use This Skill

Use for features that add or modify:
- `ornn-bench-infra/toolchain.env`
- `ornn-bench-infra/provision.sh`
- `ornn-bench-infra/benchmark_node.sh`
- `ornn-bench-infra/provider-notes/*`

## Work Procedure

1. Read `mission.md`, `validation-contract.md`, and `AGENTS.md` in the missionDir before changing scripts.
2. Preserve the toolchain-lock invariant:
   - `toolchain.env` is checked in and **must not be modified at runtime**.
   - Discovered paths go to `/opt/ornn-bench-tools/runtime.env`.
3. Make scripts deterministic and safe:
   - Use `set -euo pipefail`.
   - Never echo or persist `ORNN_API_KEY`.
   - Prefer explicit output paths.
4. Add/adjust tests:
   - Add lightweight tests that validate file presence, required keys, and forbidden patterns (e.g., writing secrets, mutating toolchain.env).
   - Tests must pass on non-GPU hosts.
5. Run validators from `.factory/services.yaml` (`test`, `lint`, `typecheck`) before finishing.

## Example Handoff

```json
{
  "salientSummary": "Added ornn-bench-infra toolchain lock + provisioning and node runner scripts with SXM5 gate and strict pin verification; added static tests to enforce immutability and secret-safety.",
  "whatWasImplemented": "Created ornn-bench-infra/{toolchain.env,provision.sh,benchmark_node.sh} and provider-notes placeholders. provision.sh installs pinned ornn-bench/torch/nvbandwidth/nccl-tests and writes /opt/ornn-bench-tools/runtime.env. benchmark_node.sh verifies pins, enforces 8xH100 SXM5 topology gate, runs 3 benchmark runs with cooldown, and stores reports/logs.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "./.venv/bin/python -m pytest -q", "exitCode": 0, "observation": "All tests passed (static infra tests included)."},
      {"command": "./.venv/bin/python -m ruff check .", "exitCode": 0, "observation": "No lint issues."},
      {"command": "./.venv/bin/python -m mypy src api", "exitCode": 0, "observation": "Typecheck passed."}
    ],
    "interactiveChecks": [
      {"action": "Open ornn-bench-infra/toolchain.env", "observed": "Contains required pins and warns it is immutable at runtime."}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/infra/test_infra_scripts.py", "cases": [{"name": "toolchain_env_has_required_keys", "verifies": "toolchain.env contains all required pin keys and full SHAs"}]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Any ambiguity about pin values, output folder conventions, or what constitutes the SXM5 gate.
- If scripts require provider-specific privileged steps that cannot be encoded safely.
- If adding these scripts would require altering mission boundaries (e.g., needing extra services).
