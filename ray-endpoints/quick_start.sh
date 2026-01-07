#!/bin/bash
# Quick start script to deploy all endpoints

set -e

echo "🚀 Starting Ray Serve endpoints deployment..."

# Check if Ray is available
if ! command -v ray &> /dev/null; then
    echo "❌ Ray not found. Make sure Ray is in PATH."
    exit 1
fi

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start Ray if not already running
if ! ray status &> /dev/null; then
    echo "📡 Starting Ray cluster..."
    ray start --head --port=6379 || echo "⚠ Ray may already be running"
fi

# Deploy endpoints
echo "📦 Deploying endpoints..."
# Run deploy.py from the ray-endpoints directory
cd "$SCRIPT_DIR"
python3 deploy.py

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Endpoints available:"
echo "  - Stable Diffusion: http://localhost:8000/api/v1/stable-diffusion"
echo "  - Kandinsky3: http://localhost:8000/api/v1/kandinsky3"
echo ""
echo "Test with:"
echo "  curl http://localhost:8000/api/v1/stable-diffusion/health"
echo "  curl http://localhost:8000/api/v1/kandinsky3/health"
