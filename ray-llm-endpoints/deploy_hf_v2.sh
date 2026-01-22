#!/bin/bash
# Wrapper script to deploy HuggingFace LLM endpoints (v2: separate apps per model)
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
    echo "   Or pass --hf-token to deploy_hf_models_v2.py"
fi

# Model names (default: DeepSeek-7B and Mistral-7B)
# Both are public models that don't require gated access
MODEL1="${MODEL1:-deepseek-ai/deepseek-llm-7b-chat}"
MODEL2="${MODEL2:-mistralai/Mistral-7B-Instruct-v0.1}"  # Public model, no token needed

# Check if user wants to deploy both models
DEPLOY_BOTH="${DEPLOY_BOTH:-true}"

if [ "$DEPLOY_BOTH" = "true" ]; then
    echo "🚀 Deploying both HuggingFace LLM models (separate applications)..."
    echo "   Model 1: $MODEL1 (DeepSeek-7B, 4096 context)"
    echo "   Model 2: $MODEL2 (Mistral-7B, 8192 context)"
    if [ -n "$HF_TOKEN" ]; then
        echo "   Using HuggingFace token"
    fi
    
    # Deploy first model
    echo ""
    echo "📦 Deploying Model 1: $MODEL1..."
    "$RAY_PYTHON" deploy_hf_models_v2.py \
        --model "$MODEL1" \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
        "$@"
    
    # Wait a bit for first deployment to complete
    sleep 3
    
    # Deploy second model (separate application)
    echo ""
    echo "📦 Deploying Model 2: $MODEL2..."
    "$RAY_PYTHON" deploy_hf_models_v2.py \
        --model "$MODEL2" \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
        "$@"
    
    echo ""
    echo "✅ Both models deployed as separate applications!"
    echo ""
    echo "📊 Endpoints:"
    APP1="hf-$(echo $MODEL1 | sed 's/\//-/g' | sed 's/_/-/g')"
    APP2="hf-$(echo $MODEL2 | sed 's/\//-/g' | sed 's/_/-/g')"
    # Get actual head node IP
    HEAD_IP=$("$RAY_PYTHON" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(ray.util.get_node_ip_address())" 2>/dev/null || echo "172.26.92.232")
    echo "   DeepSeek-7B: http://${HEAD_IP}:9202/${APP1}/v1"
    echo "   Mistral-7B:   http://${HEAD_IP}:9202/${APP2}/v1"
    echo ""
    echo "💡 Features:"
    echo "   - Separate applications (no overwriting)"
    echo "   - Autoscaling (scale to zero after 5 min idle)"
    echo "   - Mistral has 2x larger context (8192 vs 4096 tokens)"
    echo "   - Models cached on /data/models (NFS)"
else
    # Single model deployment (backward compatible)
    MODEL_NAME="${MODEL_NAME:-$MODEL1}"
    echo "🚀 Deploying HuggingFace LLM endpoint..."
    echo "   Model: $MODEL_NAME"
    if [ -n "$HF_TOKEN" ]; then
        echo "   Using HuggingFace token"
    fi
    
    "$RAY_PYTHON" deploy_hf_models_v2.py \
        --model "$MODEL_NAME" \
        ${HF_TOKEN:+--hf-token "$HF_TOKEN"} \
        --compute-report \
        "$@"
fi
