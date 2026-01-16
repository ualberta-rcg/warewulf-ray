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

# Model names (default: DeepSeek-7B and GPT-OSS-20B)
MODEL1="${MODEL1:-deepseek-ai/deepseek-llm-7b-chat}"
MODEL2="${MODEL2:-openai-community/gpt-oss-20b}"

# Check if user wants to deploy both models
DEPLOY_BOTH="${DEPLOY_BOTH:-true}"

if [ "$DEPLOY_BOTH" = "true" ]; then
    echo "🚀 Deploying both HuggingFace LLM models..."
    echo "   Model 1: $MODEL1 (DeepSeek-7B)"
    echo "   Model 2: $MODEL2 (GPT-OSS-20B)"
    if [ -n "$HF_TOKEN" ]; then
        echo "   Using HuggingFace token"
    fi
    
    # Deploy first model
    echo ""
    echo "📦 Deploying Model 1: $MODEL1..."
    "$RAY_PYTHON" deploy_hf_models.py \
        --model "$MODEL1" \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
        "$@"
    
    # Wait a bit for first deployment to complete
    sleep 2
    
    # Deploy second model (GPT-OSS) on /v2 route
    echo ""
    echo "📦 Deploying Model 2: $MODEL2..."
    "$RAY_PYTHON" deploy_vllm.py \
        --model "$MODEL2" \
        "$@"
    
    echo ""
    echo "✅ Both models deployed!"
    echo "   DeepSeek-7B: http://<head-node-ip>:8000/v1"
    echo "   GPT-OSS-20B: http://<head-node-ip>:8000/v1 (model: my-gpt-oss)"
else
    # Single model deployment (backward compatible)
    MODEL_NAME="${MODEL_NAME:-$MODEL1}"
    echo "🚀 Deploying HuggingFace LLM endpoint..."
    echo "   Model: $MODEL_NAME"
    if [ -n "$HF_TOKEN" ]; then
        echo "   Using HuggingFace token"
    fi
    
    "$RAY_PYTHON" deploy_hf_models.py \
        --model "$MODEL_NAME" \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
        "$@"
fi
