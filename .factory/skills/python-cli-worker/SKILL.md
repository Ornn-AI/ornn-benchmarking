---
name: python-cli-worker
description: Implements and verifies Python CLI benchmark orchestration, parsing, scoring, and terminal UX.
---

# Python CLI Worker

NOTE: Startup/cleanup are handled by worker-base. This skill defines feature work procedure.

## When to Use This Skill

Use for features touching `ornn-bench` CLI commands, runbook orchestration, benchmark output parsing, scoring, JSON report generation, and terminal presentation.

## Work Procedure

1. Read `mission.md`, `validation-contract.md`, `AGENTS.md`, and related `.factory/library/*.md` notes before coding.
2. Implement TDD strictly: add failing pytest coverage first for each new CLI behavior and parser rule, then implement code to pass.
3. For benchmark integrations, implement deterministic parsers using fixture outputs for mamf-finder, nvbandwidth, and nccl-tests.
4. Preserve robust UX: clear progress, clear error messages, no raw tracebacks, explicit status fields in JSON output.
5. Run targeted tests first, then full validators from `.factory/services.yaml` commands (`lint`, `typecheck`, `test`).
6. Perform manual CLI checks for touched flows (help text, command output, report rendering). Capture exact observed output.
7. If external GPU tools are unavailable, use fixtures/mocks and explicitly document limitations in handoff.

## Example Handoff

```json
{
  "salientSummary": "Implemented selective run scopes and partial-failure status propagation for the CLI run pipeline. Added parser fixtures for nvbandwidth and nccl text outputs and verified clean non-TTY output behavior.",
  "whatWasImplemented": "Added run scope filtering (`--compute-only`, `--memory-only`, `--interconnect-only`) and report manifest status fields so skipped/failed sections are explicit. Updated result serializer and scorecard rendering to show skipped/timeout states without traceback leakage.",
  "whatWasLeftUndone": "Could not perform live GPU execution because host lacks NVIDIA tooling; behavior validated via fixtures and mocked subprocess outputs.",
  "verification": {
    "commandsRun": [
      {
        "command": "python3 -m pytest tests/cli/test_run_scopes.py -q",
        "exitCode": 0,
        "observation": "All scope selection tests passed."
      },
      {
        "command": "python3 -m pytest -q",
        "exitCode": 0,
        "observation": "Full test suite passed."
      },
      {
        "command": "python3 -m ruff check .",
        "exitCode": 0,
        "observation": "No lint violations."
      }
    ],
    "interactiveChecks": [
      {
        "action": "Run `python3 -m ornn_bench --help`",
        "observed": "Help output lists run/info/report/upload commands."
      },
      {
        "action": "Run `python3 -m ornn_bench report tests/fixtures/sample_report.json`",
        "observed": "Report renders score table successfully on non-GPU host."
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/cli/test_run_scopes.py",
        "cases": [
          {
            "name": "compute_only_runs_only_compute_section",
            "verifies": "Requested section filtering is respected and non-requested sections are marked skipped."
          },
          {
            "name": "failed_section_does_not_abort_remaining_sections",
            "verifies": "Partial failures persist status and continue eligible sections."
          }
        ]
      }
    ]
  },
  "discoveredIssues": [
    {
      "severity": "medium",
      "description": "Fixture output from older nccl-tests format omits busbw column in one sample.",
      "suggestedFix": "Add format-version detection fallback parser with explicit warning."
    }
  ]
}
```

## When to Return to Orchestrator

- Benchmark behavior depends on unspecified scoring or aggregation rules.
- Required external tool semantics conflict with current parser assumptions.
- A feature requires API contract changes not yet implemented by backend worker.
- Validation failures indicate cross-cutting architecture decisions beyond this feature scope.
