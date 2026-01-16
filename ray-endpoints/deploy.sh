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

# Run the deployment script
echo "🚀 Deploying Ray Serve endpoints..."
"$RAY_PYTHON" deploy.py "$@"
