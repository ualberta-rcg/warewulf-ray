#!/bin/bash
# Wrapper script to deploy Ray Serve endpoints
# Ensures we use the correct Python with Ray installed

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Ray Python exists
RAY_PYTHON="/opt/ray/bin/python"
if [ ! -f "$RAY_PYTHON" ]; then
    echo "❌ Ray Python not found at $RAY_PYTHON"
    echo "   Make sure you're running in the Docker container"
    exit 1
fi

# Ensure PATH includes Ray
export PATH="/opt/ray/bin:$PATH"

# Check if Ray is available
if ! "$RAY_PYTHON" -c "import ray" 2>/dev/null; then
    echo "❌ Ray is not available in $RAY_PYTHON"
    exit 1
fi

# Install dependencies if needed (in Ray's environment)
# Note: Runtime environments in deployments will install deps on workers automatically
# But we also install here for head node and to fix version conflicts
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating dependencies in Ray environment..."
    echo "   (Note: Dependencies will also be installed on workers via runtime_env)"
    "$RAY_PYTHON" -m pip install --upgrade -q -r requirements.txt || {
        echo "⚠️  Warning: Some dependencies may have failed to install"
        echo "   This is OK - runtime_env will install them on workers"
    }
fi

# Install requests if needed (for Triton health check)
"$RAY_PYTHON" -m pip install -q requests 2>/dev/null || true

# Check if Triton binary is available (try NVIDIA's binary first, then nvidia-pytriton)
TRITON_BINARY="/opt/tritonserver/bin/tritonserver"
if [ ! -f "$TRITON_BINARY" ]; then
    TRITON_BINARY="/opt/ray/lib/python3.10/site-packages/pytriton/tritonserver/bin/tritonserver"
fi
if [ -f "$TRITON_BINARY" ]; then
    echo "✅ Triton binary found: $TRITON_BINARY"
    echo "   Using standalone Triton binary (recommended for Warewulf with ubuntu:24.04)"
    
    # Check if Triton is running
    if curl -s "http://localhost:8000/v2/health/live" > /dev/null 2>&1; then
        echo "✅ Triton server is already running"
    else
        echo "⚠️  Triton server not running - will be started by deploy.py if needed"
        echo "   Or start manually: bash start_triton.sh"
    fi
else
    echo "⚠️  Triton binary not found at $TRITON_BINARY"
    echo "   Make sure nvidia-pytriton is installed in Ray environment"
    echo "   (This should be installed in the Dockerfile)"
fi

# Run the deployment script
echo "🚀 Deploying Ray Serve endpoints..."
"$RAY_PYTHON" deploy.py "$@"
