#!/bin/bash
# Wrapper script to deploy HuggingFace LLM endpoint
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

# Get HuggingFace token from environment or prompt
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  No HuggingFace token found in environment"
    echo "   Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable"
    echo "   Or pass --hf-token to deploy_hf_models.py"
fi

# Model name (default: DeepSeek-7B)
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-llm-7b-chat}"

# Run the deployment script
echo "🚀 Deploying HuggingFace LLM endpoint..."
echo "   Model: $MODEL_NAME"
if [ -n "$HF_TOKEN" ]; then
    echo "   Using HuggingFace token"
fi

"$RAY_PYTHON" deploy_hf_models.py \
    --model "$MODEL_NAME" \
    ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
    "$@"
