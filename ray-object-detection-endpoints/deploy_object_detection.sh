#!/bin/bash
# Deploy Object Detection model (YOLO, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-yolov8n.pt}"
APP_NAME="${2:-}"

echo "🚀 Deploying Object Detection model: $MODEL"

/opt/ray/bin/python deploy_object_detection_v2.py \
  --model "$MODEL" \
  ${APP_NAME:+--app-name "$APP_NAME"}

echo "✅ Deployment complete!"
