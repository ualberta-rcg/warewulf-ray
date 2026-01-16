"""
Ray Serve LLM deployment for GPT-OSS models
Based on: https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm/gpt-oss/README.html
"""

import sys
import os

# Ensure we use Ray from /opt/ray
if "/opt/ray/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

# Import Ray Serve LLM components
try:
    from ray.serve.llm import LLMConfig, build_openai_app
    RAY_SERVE_LLM_AVAILABLE = True
except ImportError:
    RAY_SERVE_LLM_AVAILABLE = False
    print("⚠️  ray.serve.llm not available - may need Ray Serve LLM extension")
    print("   Install with: pip install 'ray[serve-llm]' or use Ray 2.49.0+")

# Configuration for GPT-OSS models
# You can customize these settings based on your GPU resources

def create_gpt_oss_app(model_name: str = "gpt-oss-20b", model_path: str = None):
    """
    Create a Ray Serve LLM application for GPT-OSS
    
    Args:
        model_name: Name of the model (gpt-oss-20b or gpt-oss-120b)
        model_path: Optional path to model weights (if None, downloads from HuggingFace)
    """
    
    if not RAY_SERVE_LLM_AVAILABLE:
        raise ImportError(
            "ray.serve.llm not available. "
            "Install Ray Serve LLM: pip install 'ray[serve-llm]' or use Ray 2.49.0+"
        )
    
    # Default model paths from HuggingFace
    model_paths = {
        "gpt-oss-20b": "openai-community/gpt-oss-20b",
        "gpt-oss-120b": "openai-community/gpt-oss-120b",
    }
    
    # Use provided path or default
    if model_path is None:
        model_path = model_paths.get(model_name, model_paths["gpt-oss-20b"])
    
    # LLM configuration
    # Adjust these based on your GPU resources
    # For gpt-oss-20b: typically needs 1x L4 or similar
    # For gpt-oss-120b: typically needs 2x L40S or similar
    llm_config = LLMConfig(
        model_id=model_path,
        model_task="text-generation",
        # GPU configuration - adjust based on your setup
        num_gpus=1,  # Adjust based on your GPU setup
        # Enable LLM metrics for monitoring
        log_engine_metrics=True,
        # Optional: For larger models, configure tensor parallelism
        # tensor_parallel_size=2,  # For multi-GPU setups
    )
    
    # Build OpenAI-compatible app
    # This creates an Application that provides OpenAI-compatible API
    app = build_openai_app({"llm_configs": [llm_config]})
    
    return app

# Default app for serve run/deploy
app = create_gpt_oss_app()

if __name__ == "__main__":
    # For testing/debugging
    print(f"✅ GPT-OSS application configured")
    print(f"   Model: gpt-oss-20b")
    print(f"   Deploy with: serve run serve_gpt_oss:app")
    print(f"   Or use: python deploy.py")
