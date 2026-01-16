#!/bin/bash
# Quick start script to deploy all endpoints
# Note: Ray should already be running via systemd service

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

# Check if RAY_ADDRESS is set (remote cluster)
if [ -n "$RAY_ADDRESS" ]; then
    echo "📡 RAY_ADDRESS is set to: $RAY_ADDRESS"
    echo "   Connecting to remote Ray cluster..."
    
    # Test connection
    if ! ray status --address="$RAY_ADDRESS" &> /dev/null; then
        echo "❌ Cannot connect to Ray cluster at $RAY_ADDRESS"
        echo "   Make sure the cluster is running and accessible"
        exit 1
    fi
    echo "✓ Connected to remote Ray cluster"
else
    # Check if local Ray is running (via systemd)
    if ! ray status &> /dev/null; then
        echo "❌ Ray is not running!"
        echo "   Start Ray with: systemctl start ray-worker (or ray-head)"
        echo "   Or set RAY_ADDRESS to connect to a remote cluster"
        exit 1
    else
        echo "✓ Ray cluster is running (via systemd)"
    fi
fi

# Deploy endpoints
echo "📦 Deploying endpoints..."
# Run deploy.py from the ray-endpoints directory
cd "$SCRIPT_DIR"

# Use the Ray Python from venv if available
if [ -f "/opt/ray/bin/python" ]; then
    PYTHON="/opt/ray/bin/python"
else
    PYTHON="python3"
fi

$PYTHON deploy.py

echo ""
echo "✅ Deployment complete!"
echo ""

# Get the head node IP address for network access
HEAD_IP=$(hostname -I | awk '{print $1}')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="localhost"
fi

echo "Endpoints available:"
echo "  - Stable Diffusion:"
echo "    Local:  http://localhost:8000/api/v1/stable-diffusion"
echo "    Network: http://${HEAD_IP}:8000/api/v1/stable-diffusion"
echo "  - Kandinsky3:"
echo "    Local:  http://localhost:8000/api/v1/kandinsky3"
echo "    Network: http://${HEAD_IP}:8000/api/v1/kandinsky3"
echo ""
echo "Test with:"
echo "  curl http://localhost:8000/api/v1/stable-diffusion/health"
echo "  curl http://${HEAD_IP}:8000/api/v1/stable-diffusion/health"
echo "  curl http://localhost:8000/api/v1/kandinsky3/health"
echo "  curl http://${HEAD_IP}:8000/api/v1/kandinsky3/health"
