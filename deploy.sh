#!/usr/bin/env bash
set -euo pipefail

# ── CONFIGURE THESE BEFORE RUNNING ──────────────────────────────────────────
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
SERVICE_NAME="chatbot-rag"
SERVICE_ACCOUNT="rss-feed-runner@${PROJECT_ID}.iam.gserviceaccount.com"
# ─────────────────────────────────────────────────────────────────────────────

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/rss-feed-repo/$SERVICE_NAME"

echo "==> Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

echo "==> Submitting build to Cloud Build..."
gcloud builds submit \
  --tag "$IMAGE" \
  --project "$PROJECT_ID"

echo "==> Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --service-account "$SERVICE_ACCOUNT" \
  --set-secrets "OPENAI_API_KEY=OPENAI_API_KEY:latest" \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --allow-unauthenticated \
  --project "$PROJECT_ID"

echo "==> Done. Service URL:"
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --format "value(status.url)"
