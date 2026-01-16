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
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating dependencies..."
    "$RAY_PYTHON" -m pip install -q -r requirements.txt || true
fi

# Run the deployment script
echo "🚀 Deploying Ray Serve endpoints..."
"$RAY_PYTHON" deploy.py "$@"
