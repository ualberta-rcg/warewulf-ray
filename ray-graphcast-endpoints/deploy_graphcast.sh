#!/bin/bash
# Deploy GraphCast weather forecasting model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-shermansiu/dm_graphcast}"
APP_NAME="${2:-}"

echo "🚀 Deploying GraphCast model: $MODEL"

/opt/ray/bin/python deploy_graphcast_v2.py \
  --model "$MODEL" \
  ${APP_NAME:+--app-name "$APP_NAME"}

echo "✅ Deployment complete!"
