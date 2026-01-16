#!/bin/bash
# Start Triton Inference Server
# This script starts Triton server pointing to the model repository

set -e

# Try NVIDIA's Triton binary first (from multi-stage build), fallback to nvidia-pytriton
TRITON_BINARY="/opt/tritonserver/bin/tritonserver"
if [ ! -f "$TRITON_BINARY" ]; then
    TRITON_BINARY="/opt/ray/lib/python3.10/site-packages/pytriton/tritonserver/bin/tritonserver"
fi
MODEL_REPOSITORY="/data/ray-triton/model_repository"
HTTP_PORT=8000
GRPC_PORT=8001
METRICS_PORT=8002

# Check if Triton binary exists
if [ ! -f "$TRITON_BINARY" ]; then
    echo "❌ Triton server binary not found at $TRITON_BINARY"
    echo "   Make sure you're running in the Docker container"
    exit 1
fi

# Check if model repository exists
if [ ! -d "$MODEL_REPOSITORY" ]; then
    echo "❌ Model repository not found at $MODEL_REPOSITORY"
    exit 1
fi

# Check if Triton is already running
if curl -s "http://localhost:${HTTP_PORT}/v2/health/live" > /dev/null 2>&1; then
    echo "✅ Triton server is already running on port ${HTTP_PORT}"
    exit 0
fi

echo "🚀 Starting Triton Inference Server..."
echo "   Model Repository: $MODEL_REPOSITORY"
echo "   HTTP Port: $HTTP_PORT"
echo "   gRPC Port: $GRPC_PORT"
echo "   Metrics Port: $METRICS_PORT"
echo ""

# Start Triton server
exec "$TRITON_BINARY" \
    --model-repository="$MODEL_REPOSITORY" \
    --http-port="$HTTP_PORT" \
    --grpc-port="$GRPC_PORT" \
    --metrics-port="$METRICS_PORT" \
    --log-verbose=0 \
    --strict-model-config=false \
    --exit-on-error=false
