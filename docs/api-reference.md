# API Reference

The Ornn Benchmarking API is a FastAPI service deployed on Google Cloud Run. It receives, stores, and verifies GPU benchmark results submitted by the `ornn-bench` CLI.

**Base URL:** `https://ornn-api-<hash>.run.app` (your Cloud Run service URL)

---

## Authentication

All endpoints except `/health` require an API key via the `X-API-Key` header.

```bash
curl -H "X-API-Key: YOUR_KEY" https://your-service-url/api/v1/runs
```

### Key Management

API keys are configured via the `ORNN_API_KEYS` environment variable on the server (comma-separated). Revoked keys are tracked in `ORNN_REVOKED_API_KEYS`.

| Variable | Description | Default |
|----------|-------------|---------|
| `ORNN_API_KEYS` | Comma-separated valid API keys | `dev-test-key` |
| `ORNN_REVOKED_API_KEYS` | Comma-separated revoked keys | (empty) |

### Error Responses

| Scenario | Status | Response |
|----------|--------|----------|
| Missing `X-API-Key` header | `401` | `{"detail": "Missing API key. Provide a valid key via the X-API-Key header."}` |
| Invalid API key | `401` | `{"detail": "Unauthorized."}` |
| Revoked API key | `401` | `{"detail": "Unauthorized."}` |

> **Security:** Error messages never reveal which keys exist, are revoked, or why a specific key was rejected.

---

## Endpoints

### `GET /health`

Health check / liveness probe. No authentication required.

**Response** `200 OK`:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "service": "Ornn Benchmarking API"
}
```

---

### `POST /api/v1/runs`

Submit a benchmark run. Idempotent — duplicate submissions (same `report_id` + `created_at` + `schema_version`) return the existing run without creating duplicates.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Valid API key |
| `Content-Type` | Yes | `application/json` |

**Request Body (`RunPayload`):**

```json
{
  "schema_version": "1.0.0",
  "report_id": "abc123",
  "created_at": "2025-01-15T10:30:00Z",
  "system_inventory": {
    "gpus": [
      {
        "uuid": "GPU-abc-123",
        "name": "NVIDIA H100",
        "driver_version": "535.129.03",
        "cuda_version": "12.2",
        "memory_total_mb": 81920
      }
    ],
    "os_info": "Linux 5.15.0",
    "kernel_version": "5.15.0",
    "cpu_model": "Intel Xeon w9-3495X",
    "numa_nodes": 2,
    "pytorch_version": "2.1.0"
  },
  "sections": [
    {
      "name": "compute",
      "status": "completed",
      "started_at": "2025-01-15T10:30:05Z",
      "finished_at": "2025-01-15T10:35:00Z",
      "metrics": {"bf16_tflops": 989.5, "fp8_tflops": 1978.2},
      "error": null
    }
  ],
  "scores": {
    "ornn_i": 95.2,
    "ornn_t": 88.7,
    "qualification": "Premium",
    "components": {"bw": 3.35, "fp8": 1978.2, "bf16": 989.5, "ar": 450.0},
    "score_status": "valid",
    "score_status_detail": null,
    "aggregate_method": "minimum"
  },
  "manifest": {}
}
```

**Required Fields:**

| Field | Type | Constraints |
|-------|------|-------------|
| `schema_version` | `string` | Non-empty. Must be a supported version (currently `1.0.0`) |
| `report_id` | `string` | Non-empty. Client-generated unique identifier |
| `created_at` | `string` | Non-empty. ISO-8601 timestamp |
| `system_inventory` | `object` | System inventory payload |
| `scores` | `object` | Computed Ornn scores |

**Response — New Run** `201 Created`:

```json
{
  "run_id": "a1b2c3d4e5f6...",
  "received_at": "2025-01-15T10:35:05Z",
  "stored_at": "2025-01-15T10:35:05Z"
}
```

**Response — Duplicate** `200 OK`:

```json
{
  "run_id": "a1b2c3d4e5f6...",
  "received_at": "2025-01-15T10:35:05Z",
  "stored_at": "2025-01-15T10:35:05Z"
}
```

The duplicate detection uses a SHA-256 hash of `report_id` + `created_at` + `schema_version` to deterministically identify identical logical runs.

---

### `GET /api/v1/runs/{run_id}`

Retrieve a stored benchmark run by its server-assigned ID.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Valid API key |

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | `string` | Server-assigned run identifier from submission |

**Response** `200 OK`:

Returns the full stored document including the original payload fields plus server metadata (`run_id`, `received_at`, `stored_at`, `dedupe_key`).

**Error Responses:**

| Status | Condition |
|--------|-----------|
| `401` | Missing or invalid API key |
| `404` | `{"detail": "Run not found."}` |
| `429` | Rate limit exceeded |

---

### `POST /api/v1/verify`

Verify client-computed scores against server-side recomputation. The server recomputes Ornn-I and Ornn-T from the raw metric components and compares against the submitted values.

**Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Valid API key |
| `Content-Type` | Yes | `application/json` |

**Request Body (`VerifyRequest`):**

```json
{
  "components": {
    "bw": 3.35,
    "fp8": 1978.2,
    "bf16": 989.5,
    "ar": 450.0
  },
  "ornn_i": 95.2,
  "ornn_t": 88.7,
  "qualification": "Premium"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `components` | `dict[str, float]` | Raw metric values (`bw`, `fp8`, `bf16`, `ar`) |
| `ornn_i` | `float \| null` | Client-computed Ornn-I score |
| `ornn_t` | `float \| null` | Client-computed Ornn-T score |
| `qualification` | `string \| null` | Client-computed qualification grade |

**Response — Verified** `200 OK`:

```json
{
  "status": "verified",
  "server_ornn_i": 95.2,
  "server_ornn_t": 88.7,
  "server_qualification": "Premium",
  "metric_details": [
    {
      "metric": "ornn_i",
      "submitted": 95.2,
      "server_computed": 95.2,
      "match": true,
      "delta": 0.0
    },
    {
      "metric": "ornn_t",
      "submitted": 88.7,
      "server_computed": 88.7,
      "match": true,
      "delta": 0.0
    },
    {
      "metric": "qualification",
      "submitted": null,
      "server_computed": null,
      "match": true,
      "delta": null
    }
  ],
  "tolerance": 0.01
}
```

**Response — Mismatch** `200 OK`:

When submitted and server-computed scores differ beyond the tolerance threshold, the response includes `"status": "mismatch"` with per-metric details showing exactly which values diverged and by how much.

---

## Request/Response Schemas

### System Inventory

| Field | Type | Description |
|-------|------|-------------|
| `gpus` | `GPUInfoPayload[]` | List of GPU details |
| `os_info` | `string` | Operating system description |
| `kernel_version` | `string` | Kernel version string |
| `cpu_model` | `string` | CPU model name |
| `numa_nodes` | `int` | Number of NUMA nodes |
| `pytorch_version` | `string` | PyTorch version |

### GPU Info

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `string` | GPU UUID |
| `name` | `string` | GPU model name |
| `driver_version` | `string` | NVIDIA driver version |
| `cuda_version` | `string` | CUDA version |
| `memory_total_mb` | `int` | Total GPU memory in MB |

### Score Payload

| Field | Type | Description |
|-------|------|-------------|
| `ornn_i` | `float \| null` | Ornn-I inference score |
| `ornn_t` | `float \| null` | Ornn-T training score |
| `qualification` | `string \| null` | `Premium`, `Standard`, or `Below` |
| `components` | `dict[str, float]` | Raw component metrics |
| `score_status` | `string` | `valid`, `partial`, or `error` |
| `score_status_detail` | `string \| null` | Human-readable detail for non-valid status |
| `aggregate_method` | `string \| null` | Aggregation method for multi-GPU |

### Section Payload

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Section name (e.g., `compute`, `memory`) |
| `status` | `string` | `pending`, `running`, `completed`, `failed`, `skipped`, `timeout` |
| `started_at` | `string \| null` | ISO-8601 start timestamp |
| `finished_at` | `string \| null` | ISO-8601 end timestamp |
| `metrics` | `dict` | Section-specific metric data |
| `error` | `string \| null` | Error message if failed |

---

## Error Codes

| HTTP Status | Meaning | When |
|-------------|---------|------|
| `200` | Success / Duplicate | Successful retrieval or duplicate submission |
| `201` | Created | New run successfully stored |
| `401` | Unauthorized | Missing, invalid, or revoked API key |
| `404` | Not Found | Run ID does not exist |
| `422` | Unprocessable Entity | Malformed payload or unsupported schema version |
| `429` | Too Many Requests | Rate limit exceeded |

### Schema Version Errors

Unsupported schema versions return `422` with upgrade/downgrade guidance:

```json
{
  "detail": "Unsupported schema version '2.0.0'. Supported versions: 1.0.0. Please upgrade or downgrade ornn-bench to a compatible version."
}
```

### Validation Errors

Malformed payloads return `422` with field-level details:

```json
{
  "detail": [
    {
      "loc": ["body", "report_id"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

---

## Rate Limiting

The API enforces per-key rate limits using a sliding-window counter.

| Parameter | Default | Environment Variable |
|-----------|---------|---------------------|
| Max requests | 60 | `RATE_LIMIT_REQUESTS` |
| Window | 60 seconds | `RATE_LIMIT_WINDOW_SECONDS` |

When the limit is exceeded, the API returns:

- **Status:** `429 Too Many Requests`
- **Header:** `Retry-After: <seconds>` indicating when to retry
- **Body:** `{"detail": "Rate limit exceeded. Please retry later."}`

---

## Idempotent Ingest

The API implements idempotent ingest to prevent duplicate benchmark runs:

1. A **dedupe key** is computed as `SHA-256(report_id + created_at + schema_version)`
2. Before creating a new document, the API checks Firestore for an existing document with the same dedupe key
3. If found, returns the existing run with `200 OK` instead of `201 Created`
4. CLI retries after network failures are always safe — no duplicates will be created

---

## Client Configuration

The `ornn-bench` CLI connects to the API using these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ORNN_API_URL` | API base URL | `https://ornn-benchmarking-api.run.app` |
| `ORNN_API_KEY` | API key for authentication | (none — required for upload/verify) |
