#!/bin/bash
# Quick start script to deploy Triton-based endpoints
# These are standalone deployments that work better than the BaseEndpoint ones

set -e

echo "🚀 Starting Ray Serve Triton endpoints deployment..."

# Check if cleanup flag is set
CLEANUP=${1:-""}
if [ "$CLEANUP" = "--cleanup" ] || [ "$CLEANUP" = "-c" ]; then
    echo "🧹 Cleaning up before deployment..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/cleanup_ray_serve.sh" ]; then
        bash "$SCRIPT_DIR/cleanup_ray_serve.sh"
        echo ""
        sleep 2
    else
        echo "⚠ Cleanup script not found, skipping cleanup"
    fi
fi

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

# Use the Ray Python from venv if available
if [ -f "/opt/ray/bin/python" ]; then
    PYTHON="/opt/ray/bin/python"
else
    PYTHON="python3"
fi

# Check for shared venv and install dependencies if needed
SHARED_VENV="/data/ray-endpoints-venv"
if [ -d "$SHARED_VENV" ] && [ -f "$SHARED_VENV/bin/pip" ]; then
    PIP="$SHARED_VENV/bin/pip"
    echo "✓ Using shared venv: $SHARED_VENV"
else
    PIP="/opt/ray/bin/pip"
    echo "⚠ Using Ray venv: /opt/ray"
fi

# Check and install dependencies
echo "📦 Checking dependencies..."
MISSING_DEPS=()
for pkg in "fastapi" "diffusers" "torch"; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING_DEPS+=("$pkg")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "⚠ Missing dependencies: ${MISSING_DEPS[*]}"
    echo "   Installing..."
    $PIP install --quiet "${MISSING_DEPS[@]}" || {
        echo "⚠ Some dependencies failed to install, continuing anyway..."
    }
fi

# Start Ray Serve with HTTP options to listen on all interfaces
echo "📡 Starting Ray Serve..."
$PYTHON << 'EOF'
from ray import serve
from ray.serve.config import HTTPOptions
import ray
import os
import subprocess
import socket

# Connect to Ray
ray_address = os.environ.get("RAY_ADDRESS")
if ray_address:
    ray.init(address=ray_address, ignore_reinit_error=True)
else:
    # Get head node IP
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            host_ip = result.stdout.strip().split()[0]
            ray_address = f"{host_ip}:6379"
        else:
            hostname = socket.gethostname()
            host_ip = socket.gethostbyname(hostname)
            ray_address = f"{host_ip}:6379"
    except Exception:
        ray_address = "localhost:6379"
    
    ray.init(address=ray_address, ignore_reinit_error=True)

# Start Serve with HTTP options
try:
    serve.start(detached=True)
    serve.shutdown()
    print("⚠ Ray Serve already running, restarting with new HTTP config...")
    import time
    time.sleep(2)
except Exception:
    pass

try:
    http_options = HTTPOptions(host="0.0.0.0", port=8000)
    serve.start(detached=True, http_options=http_options)
    print("✓ Ray Serve started on 0.0.0.0:8000")
except Exception as e:
    try:
        serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
        print("✓ Ray Serve started on 0.0.0.0:8000")
    except Exception:
        serve.start(detached=True)
        print("⚠ Ray Serve started (may need to configure host manually)")
EOF

# Deploy Triton endpoints using the deployment script
echo ""
echo "📦 Deploying Triton endpoints..."
cd "$SCRIPT_DIR"
$PYTHON deploy_triton.py
