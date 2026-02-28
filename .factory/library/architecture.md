# Architecture

Architecture decisions and implementation patterns for this mission.

---

## High-Level Components

1. **Python CLI (`ornn-bench`)**
   - Runs full Section 8 benchmark workflow
   - Parses benchmark outputs and computes local Ornn-I/Ornn-T
   - Produces pretty terminal scorecard + JSON report
   - Optionally uploads report to backend API

2. **FastAPI Backend (`/api/v1/*`)**
   - Receives benchmark runs
   - Applies API-key auth
   - Stores/retrieves run data in Firestore
   - Recomputes/verifies scores server-side

3. **Firestore Storage**
   - Benchmark run documents
   - Dedup identity metadata
   - Verification status records

## Design Constraints

- Full Section 8 runbook coverage (compute/memory/interconnect/monitoring/post-flight)
- Deterministic parsing and scoring using fixtures for non-GPU development
- Free-tier-safe infrastructure defaults (Cloud Run + Firestore only)
- No dependency on existing Supabase projects

## Scoring Model

- Ornn-I = `55 * (BW / BW_ref) + 45 * (FP8 / FP8_ref)`
- Ornn-T = `55 * (BF16 / BF16_ref) + 45 * (AR / AR_ref)`

Qualification must apply floor checks + composite gate deterministically.
