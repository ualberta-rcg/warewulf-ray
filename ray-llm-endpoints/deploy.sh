#!/bin/bash
# Wrapper script to deploy GPT-OSS LLM endpoint
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

# Install dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/updating dependencies in Ray environment..."
    "$RAY_PYTHON" -m pip install --upgrade -q -r requirements.txt || {
        echo "⚠️  Warning: Some dependencies may have failed to install"
        echo "   This is OK if they're already installed"
    }
fi

# Run the deployment script
echo "🚀 Deploying GPT-OSS LLM endpoint..."
"$RAY_PYTHON" deploy.py "$@"
