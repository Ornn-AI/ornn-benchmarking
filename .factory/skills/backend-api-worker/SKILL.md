---
name: backend-api-worker
description: Implements and verifies FastAPI upload/verification APIs, auth, dedupe, and Firestore persistence.
---

# Backend API Worker

NOTE: Startup/cleanup are handled by worker-base. This skill defines feature work procedure.

## When to Use This Skill

Use for features in the Cloud Run FastAPI service: API routes, API-key auth, Firestore storage layer, idempotent ingest, server-side score verification, and API tests.

## Work Procedure

1. Read mission artifacts (`mission.md`, `validation-contract.md`, `AGENTS.md`) and relevant library notes before coding.
2. Add failing tests first (request validation, auth, dedupe, verify logic) using pytest + FastAPI TestClient.
3. Implement route handlers and service layer with strict schema validation and deterministic error responses.
4. Enforce security basics: never log raw API keys, sanitize error details, and reject unauthorized requests consistently.
5. Implement idempotent ingest and deterministic score verification with explicit mismatch payloads.
6. Run targeted API tests, then full validators from `.factory/services.yaml` (`lint`, `typecheck`, `test`).
7. Perform manual API checks with curl for success + failure cases and capture exact response behavior.

## Example Handoff

```json
{
  "salientSummary": "Implemented run ingest and retrieval endpoints with API-key auth and deterministic 401/422 semantics. Added server-side score verification endpoint with per-metric mismatch reporting.",
  "whatWasImplemented": "Added `/api/v1/runs` POST with payload validation and Firestore persistence, `/api/v1/runs/{id}` GET with auth checks, and `/api/v1/verify` endpoint that recomputes Ornn-I/Ornn-T and returns verified/mismatch results. Implemented deterministic dedupe key to prevent duplicate logical runs on retry.",
  "whatWasLeftUndone": "Rate-limiting middleware not included in this feature because no limiter dependency existed yet.",
  "verification": {
    "commandsRun": [
      {
        "command": "python3 -m pytest tests/api/test_runs_endpoints.py -q",
        "exitCode": 0,
        "observation": "Endpoint auth/validation/retrieval tests passed."
      },
      {
        "command": "python3 -m pytest tests/api/test_verify_endpoint.py -q",
        "exitCode": 0,
        "observation": "Verification match/mismatch fixtures passed."
      },
      {
        "command": "python3 -m pytest -q",
        "exitCode": 0,
        "observation": "Full suite passed."
      }
    ],
    "interactiveChecks": [
      {
        "action": "curl POST /api/v1/runs with valid key",
        "observed": "Received 201 with run_id and server timestamps."
      },
      {
        "action": "curl POST /api/v1/runs without key",
        "observed": "Received 401 with sanitized error response."
      },
      {
        "action": "curl POST /api/v1/verify with mismatch payload",
        "observed": "Received mismatch result including per-metric diffs."
      }
    ]
  },
  "tests": {
    "added": [
      {
        "file": "tests/api/test_runs_endpoints.py",
        "cases": [
          {
            "name": "post_runs_requires_api_key",
            "verifies": "Missing API key returns 401."
          },
          {
            "name": "post_runs_invalid_payload_returns_422",
            "verifies": "Schema validation errors are actionable."
          },
          {
            "name": "duplicate_payload_reuses_existing_logical_run",
            "verifies": "Idempotent ingest contract."
          }
        ]
      }
    ]
  },
  "discoveredIssues": [
    {
      "severity": "low",
      "description": "Firestore emulator configuration differs from production credentials flow.",
      "suggestedFix": "Document emulator toggle and CI defaults in environment library file."
    }
  ]
}
```

## When to Return to Orchestrator

- API behavior requires policy decisions (auth model, tolerance values, dedupe identity) not defined in mission artifacts.
- Cloud dependency or credentials are unavailable and block endpoint verification.
- Route changes require coordinated CLI contract changes in other pending features.
