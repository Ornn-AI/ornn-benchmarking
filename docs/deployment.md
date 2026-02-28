# Deployment Guide

This guide covers deploying the Ornn Benchmarking API to Google Cloud Run and setting up local testing environments.

## Architecture Overview

The Ornn Benchmarking API uses **only two GCP services** by design:

| Service | Purpose | Cost |
|---------|---------|------|
| **Cloud Run** | Hosts the FastAPI application | Free tier: 2M requests/month, scales to zero |
| **Firestore** | Stores benchmark run data | Free tier: 20K writes/day, 50K reads/day |

No always-on infrastructure is used. The API scales to zero when idle, resulting in **zero cost at low usage**.

### Services NOT Used (by design)

- ❌ Compute Engine VMs
- ❌ Cloud SQL
- ❌ GKE / Kubernetes
- ❌ App Engine
- ❌ Cloud Functions (legacy)
- ❌ Any always-on paid infrastructure

---

## Prerequisites

### GCP Setup

1. **GCP Project**: Create or select a project (default: `ornn-benchmarking`)
2. **Enable APIs**:
   ```bash
   gcloud services enable \
       run.googleapis.com \
       firestore.googleapis.com \
       cloudbuild.googleapis.com \
       --project=ornn-benchmarking
   ```
3. **Firestore**: Initialize in native mode in `us-east1`:
   ```bash
   gcloud firestore databases create \
       --project=ornn-benchmarking \
       --location=us-east1
   ```

### Local Tools

- `gcloud` CLI (authenticated with `gcloud auth login`)
- Docker (for local container testing, optional)

---

## Deployment

### Quick Deploy

```bash
./deploy.sh
```

This deploys with free-tier-safe defaults:
- Project: `ornn-benchmarking`
- Region: `us-east1`
- Min instances: `0` (scale-to-zero)
- Max instances: `2`

### Custom Deploy

```bash
./deploy.sh \
    --project my-project \
    --region us-central1 \
    --max-instances 5
```

### Manual Deploy (without script)

```bash
# Build the container
gcloud builds submit \
    --project=ornn-benchmarking \
    --tag=gcr.io/ornn-benchmarking/ornn-api

# Deploy to Cloud Run
gcloud run deploy ornn-api \
    --project=ornn-benchmarking \
    --region=us-east1 \
    --image=gcr.io/ornn-benchmarking/ornn-api \
    --platform=managed \
    --min-instances=0 \
    --max-instances=2 \
    --memory=256Mi \
    --set-env-vars="FIRESTORE_PROJECT_ID=ornn-benchmarking" \
    --allow-unauthenticated
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FIRESTORE_PROJECT_ID` | GCP project for Firestore | `ornn-benchmarking` |
| `PORT` | HTTP port (set by Cloud Run) | `8080` |
| `RATE_LIMIT_REQUESTS` | Max requests per key per window | `60` |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window duration | `60` |
| `DEBUG` | Enable debug mode | `false` |

### Post-Deploy Verification

```bash
# Get the service URL
SERVICE_URL=$(gcloud run services describe ornn-api \
    --project=ornn-benchmarking \
    --region=us-east1 \
    --format="value(status.url)")

# Health check
curl -sf "${SERVICE_URL}/health"

# Verify min-instances is 0 (scale-to-zero)
gcloud run services describe ornn-api \
    --project=ornn-benchmarking \
    --region=us-east1 \
    --format="value(spec.template.spec.containerConcurrency)"
```

---

## Local Development & Testing

### Option 1: Direct Python (Recommended for Development)

Run the API server locally without Docker or Firestore:

```bash
# Install dependencies
pip install -e ".[dev]"

# Start the API (uses in-memory mock storage by default when
# FIRESTORE_EMULATOR_HOST is not set and google-cloud-firestore
# client cannot connect)
PORT=8080 python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Option 2: With Firestore Emulator

The Firestore emulator provides a local, protocol-compatible Firestore instance for integration testing.

#### Java Prerequisite

> **⚠️ The Firestore emulator requires a Java runtime (JDK 11+) installed on the host machine.**
>
> If Java is not available, use Option 1 (direct Python with mock storage) or Option 3 (Docker) instead.

Check Java availability:
```bash
java -version
```

Install Java if needed:
```bash
# macOS (Homebrew)
brew install openjdk@17

# Ubuntu/Debian
sudo apt-get install openjdk-17-jdk

# Verify
java -version
```

#### Start the Emulator

```bash
# Install the emulator component
gcloud components install cloud-firestore-emulator

# Start the emulator on port 8085
gcloud emulators firestore start --host-port=127.0.0.1:8085
```

#### Run the API Against the Emulator

```bash
# In a separate terminal
FIRESTORE_EMULATOR_HOST=127.0.0.1:8085 \
PORT=8080 \
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Option 3: Mock Fallback for Tests

When neither a live Firestore instance nor the emulator is available (e.g., CI environments, machines without Java), the test suite uses an **in-memory mock Firestore client** that implements the same interface:

```bash
# Run all tests (automatically uses mock storage)
python -m pytest -q

# Run API tests specifically
python -m pytest tests/api/ -q
```

The mock client is configured in `tests/conftest.py` and provides:
- Full CRUD operations matching the Firestore API surface
- Deterministic behavior for unit and integration tests
- No external dependencies (no Java, no network, no GCP credentials)

This is the recommended path for:
- **CI/CD pipelines** (GitHub Actions)
- **Development machines without Java**
- **Quick local validation**

### Option 4: Docker Local Testing

```bash
# Build the container locally
docker build -t ornn-api .

# Run with mock storage (no Firestore connection)
docker run -p 8080:8080 -e PORT=8080 ornn-api

# Run against local Firestore emulator
docker run -p 8080:8080 \
    -e PORT=8080 \
    -e FIRESTORE_EMULATOR_HOST=host.docker.internal:8085 \
    ornn-api
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `gcloud: command not found` | Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) |
| Firestore emulator won't start | Install Java JDK 11+ (see Java Prerequisite above) |
| Permission denied on deploy | Run `gcloud auth login` and ensure project permissions |
| Cold start latency | Expected with scale-to-zero; first request may take 2-5 seconds |
| Port conflict on 8080 | Set `PORT=8081` or stop conflicting service |

### Verifying Free-Tier Compliance

After deployment, verify the configuration enforces free-tier defaults:

```bash
# Check min instances is 0 (scale-to-zero)
gcloud run services describe ornn-api \
    --project=ornn-benchmarking \
    --region=us-east1 \
    --format="yaml(spec.template.metadata.annotations)"

# The output should include:
#   autoscaling.knative.dev/minScale: '0'
```
