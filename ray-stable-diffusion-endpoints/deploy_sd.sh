#!/bin/bash
# Wrapper script to deploy Stable Diffusion endpoints
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

# Model directory
MODEL_DIR="${MODEL_DIR:-/data/models/stablediffusion}"

# Check if user wants to list models
if [ "$1" = "--list" ] || [ "$1" = "-l" ] || [ "$1" = "list" ]; then
    echo "📋 Listing available Stable Diffusion models..."
    "$RAY_PYTHON" deploy_stable_diffusion_v2.py --list-models
    exit 0
fi

# If no arguments, show usage
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model-file> [options]"
    echo "   or: $0 --list"
    echo ""
    echo "Examples:"
    echo "  $0 /data/models/stablediffusion/sd_xl_base_1.0.safetensors"
    echo "  $0 /data/models/stablediffusion/v1-5-pruned-emaonly.safetensors"
    echo "  $0 --list"
    echo ""
    echo "Options:"
    echo "  --list, -l    List available models"
    echo "  --app-name    Custom application name"
    echo ""
    echo "To list available models:"
    echo "  $0 --list"
    exit 1
fi

# Get model path (first argument)
MODEL_PATH="$1"

# Check if it's actually a file path (not a flag)
if [ "$MODEL_PATH" = "--list" ] || [ "$MODEL_PATH" = "-l" ] || [ "$MODEL_PATH" = "list" ]; then
    echo "📋 Listing available Stable Diffusion models..."
    "$RAY_PYTHON" deploy_stable_diffusion_v2.py --list-models
    exit 0
fi

# Validate model path
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo ""
    echo "Available models:"
    "$RAY_PYTHON" deploy_stable_diffusion_v2.py --list-models
    exit 1
fi

shift  # Remove first argument, pass rest to deploy script

# Get model name from path
MODEL_NAME=$(basename "$MODEL_PATH" .safetensors)

echo "🚀 Deploying Stable Diffusion endpoint..."
echo "   Model: $MODEL_NAME"
echo "   Path: $MODEL_PATH"

# Run the deployment script
"$RAY_PYTHON" deploy_stable_diffusion_v2.py \
    --model-path "$MODEL_PATH" \
    --model-name "$MODEL_NAME" \
    "$@"
