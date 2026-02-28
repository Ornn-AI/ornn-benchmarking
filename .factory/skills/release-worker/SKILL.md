---
name: release-worker
description: Finalizes packaging, CI/CD, deployment scripts, and release-readiness quality checks.
---

# Release Worker

NOTE: Startup/cleanup are handled by worker-base. This skill defines feature work procedure.

## When to Use This Skill

Use for packaging/distribution, GitHub Actions workflows, deployment automation, repo hygiene, and release-quality documentation tasks.

## Work Procedure

1. Read mission artifacts and ensure all required functionality already exists before release polishing.
2. Add tests first for release-critical logic (entry points, config parsing, workflow validation helpers) where feasible.
3. Implement packaging metadata, CLI entrypoint mapping, and reproducible install path (`pip install ornn-bench`).
4. Implement CI workflows with free-tier-safe defaults (no paid runners/services), including test/lint/typecheck gates.
5. Verify deployment scripts target only approved infrastructure (Cloud Run + Firestore).
6. Run full validators from `.factory/services.yaml` and verify install/run commands manually.
7. Ensure repository cleanliness: no temporary files, no secrets, no generated junk checked in.

## Example Handoff

```json
{
  "salientSummary": "Completed release packaging and CI workflows for public open-source launch. Verified pip install entrypoint, test gates, and Cloud Run deploy script constraints.",
  "whatWasImplemented": "Added pyproject packaging metadata, console_scripts entrypoint, GitHub Actions workflows for lint/typecheck/test and release publish path, plus Cloud Run deployment script constrained to approved services. Updated repository-level release docs and examples.",
  "whatWasLeftUndone": "PyPI publish was not executed in-session because release token is not available in local environment.",
  "verification": {
    "commandsRun": [
      {
        "command": "python3 -m pip install -e .",
        "exitCode": 0,
        "observation": "Editable install succeeded and console command was created."
      },
      {
        "command": "ornn-bench --help",
        "exitCode": 0,
        "observation": "CLI command available post-install with expected subcommands."
      },
      {
        "command": "python3 -m pytest -q && python3 -m ruff check .",
        "exitCode": 0,
        "observation": "All validators passed before handoff."
      }
    ],
    "interactiveChecks": [
      {
        "action": "Inspect GitHub Actions workflow files",
        "observed": "Workflows run only free-tier-compatible jobs and do not reference paid infrastructure."
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_cli_entrypoint.py",
        "cases": [
          {
            "name": "console_script_invokes_main_app",
            "verifies": "Packaging entrypoint is wired correctly."
          }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Required credentials (PyPI, deployment secrets, GitHub tokens) are missing for publish/deploy steps.
- CI or packaging failures indicate unresolved upstream implementation gaps.
- Infrastructure commands require adding non-approved paid services.
