#!/usr/bin/env bash
# deploy.sh — Deploy Ornn Benchmarking API to Cloud Run
#
# Free-tier-safe defaults:
#   - Cloud Run with scale-to-zero (--min-instances=0)
#   - Firestore (native mode) as the only database
#   - No paid or persistent infrastructure (VM, managed DB, etc.)
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project 'ornn-benchmarking' created
#   - Firestore in native mode enabled (us-east1)
#
# Usage:
#   ./deploy.sh                        # deploy with defaults
#   ./deploy.sh --project my-project   # override project
#
# Environment variables (optional overrides):
#   GCP_PROJECT    — GCP project ID (default: ornn-benchmarking)
#   GCP_REGION     — Cloud Run region (default: us-east1)
#   SERVICE_NAME   — Cloud Run service name (default: ornn-api)
#   MAX_INSTANCES  — Maximum Cloud Run instances (default: 2)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — free-tier-safe defaults
# ---------------------------------------------------------------------------
GCP_PROJECT="${GCP_PROJECT:-ornn-benchmarking}"
GCP_REGION="${GCP_REGION:-us-east1}"
SERVICE_NAME="${SERVICE_NAME:-ornn-api}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"

# Fixed free-tier constraints (not overridable)
MIN_INSTANCES=0          # Scale-to-zero: no cost when idle
MEMORY="256Mi"           # Minimal memory for FastAPI
CPU="1"                  # Single vCPU
CONCURRENCY="80"        # Requests per instance
TIMEOUT="60"             # Request timeout in seconds
PORT="8080"              # Container port (Cloud Run injects $PORT)
PLATFORM="managed"       # Cloud Run managed (serverless)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            GCP_PROJECT="$2"
            shift 2
            ;;
        --region)
            GCP_REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./deploy.sh [--project PROJECT] [--region REGION] [--service NAME] [--max-instances N]"
            echo ""
            echo "Deploys the Ornn Benchmarking API to Cloud Run with free-tier-safe defaults."
            echo ""
            echo "Options:"
            echo "  --project        GCP project ID (default: ornn-benchmarking)"
            echo "  --region         Cloud Run region (default: us-east1)"
            echo "  --service        Cloud Run service name (default: ornn-api)"
            echo "  --max-instances  Maximum instances (default: 2)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './deploy.sh --help' for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed."
    echo "Install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "=== Ornn Benchmarking API Deployment ==="
echo ""
echo "  Project:        ${GCP_PROJECT}"
echo "  Region:         ${GCP_REGION}"
echo "  Service:        ${SERVICE_NAME}"
echo "  Min instances:  ${MIN_INSTANCES} (scale-to-zero)"
echo "  Max instances:  ${MAX_INSTANCES}"
echo "  Memory:         ${MEMORY}"
echo "  CPU:            ${CPU}"
echo "  Platform:       Cloud Run (managed)"
echo "  Database:       Firestore (native mode)"
echo ""

# ---------------------------------------------------------------------------
# Build and deploy
# ---------------------------------------------------------------------------
echo ">>> Building container image..."
gcloud builds submit \
    --project="${GCP_PROJECT}" \
    --tag="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}" \
    .

echo ""
echo ">>> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --project="${GCP_PROJECT}" \
    --region="${GCP_REGION}" \
    --image="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}" \
    --platform="${PLATFORM}" \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --concurrency="${CONCURRENCY}" \
    --timeout="${TIMEOUT}" \
    --set-env-vars="FIRESTORE_PROJECT_ID=${GCP_PROJECT}" \
    --allow-unauthenticated \
    --quiet

echo ""
echo ">>> Deployment complete!"
echo ""

# ---------------------------------------------------------------------------
# Post-deploy verification
# ---------------------------------------------------------------------------
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${GCP_PROJECT}" \
    --region="${GCP_REGION}" \
    --format="value(status.url)")

echo "Service URL: ${SERVICE_URL}"
echo ""
echo ">>> Running health check..."
if curl -sf "${SERVICE_URL}/health"; then
    echo ""
    echo "✅ Health check passed!"
else
    echo ""
    echo "⚠️  Health check did not succeed. The service may still be starting."
    echo "    Try: curl ${SERVICE_URL}/health"
fi
