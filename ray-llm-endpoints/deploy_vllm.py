#!/usr/bin/env python3
"""
Deploy GPT-OSS LLM endpoint using vLLM directly (no ray.serve.llm needed)
"""

import argparse
import sys
import os

# Ensure we use Ray from /opt/ray
RAY_PYTHON = "/opt/ray/bin/python"
if os.path.exists(RAY_PYTHON) and __file__ != os.path.abspath(RAY_PYTHON):
    if "/opt/ray/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ray import serve
    from ray.serve.config import HTTPOptions
    import ray
except ImportError:
    print("❌ Ray not found. Please use /opt/ray/bin/python to run this script.")
    print("   Current Python: " + sys.executable)
    sys.exit(1)

# Import the vLLM-based GPT-OSS app
try:
    from serve_gpt_oss_vllm import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_gpt_oss_vllm: {e}")
    print("   Make sure serve_gpt_oss_vllm.py is in the same directory")
    sys.exit(1)


def deploy_gpt_oss(model_name: str = "openai-community/gpt-oss-20b"):
    """Deploy GPT-OSS LLM endpoint using vLLM"""
    
    print(f"🚀 Deploying GPT-OSS LLM endpoint (vLLM direct)...")
    print(f"   Model: {model_name}")
    
    # Create the application
    app = create_app(model_name=model_name)
    
    # Deploy using serve.run
    serve.run(
        app,
        route_prefix="/v1",  # OpenAI-compatible API endpoint
    )
    
    print(f"✅ GPT-OSS endpoint deployed successfully!")
    print(f"   OpenAI API: http://<head-node-ip>:8001/v1")
    print(f"   Model name: my-gpt-oss")


def main():
    parser = argparse.ArgumentParser(description="Deploy GPT-OSS LLM endpoint (vLLM direct)")
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt-oss-20b",
        help="Model to deploy (HuggingFace model ID, default: openai-community/gpt-oss-20b)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Initialize Ray connection
    try:
        ray.init(address=args.address, ignore_reinit_error=True)
        print(f"✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        sys.exit(1)
    
    # Check for vLLM
    print("🔍 Checking dependencies...")
    try:
        import vllm
        print(f"✅ vLLM available (version: {vllm.__version__})")
    except ImportError:
        print("⚠️  vLLM not found.")
        print("   Note: vLLM will be installed automatically on worker nodes via runtime_env")
        print("   But you may need to install it manually if deployment fails:")
        print("   /opt/ray/bin/pip install vllm>=0.10.1")
        print("   (Note: This requires disk space - check with: df -h)")
    
    # Start Ray Serve with HTTP options
    RAY_SERVE_PORT = 8001  # Use 8001 to avoid conflict with Triton on 8000
    try:
        # Check if Serve is already running
        try:
            status = serve.status()
            print(f"✅ Ray Serve is already running")
            print(f"   Current applications: {len(status.applications) if hasattr(status, 'applications') else 'N/A'}")
        except:
            # Serve not running, start it
            http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
            serve.start(detached=True, http_options=http_options)
            print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT}")
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Continuing anyway...")
    
    # Deploy the endpoint
    try:
        deploy_gpt_oss(model_name=args.model)
        
        print("\n📊 Deployment Summary:")
        print(f"   Model: {args.model}")
        print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/v1")
        print(f"   Model name: my-gpt-oss")
        print(f"\n📝 Example usage:")
        print(f"   curl http://<head-node-ip>:{RAY_SERVE_PORT}/v1/chat/completions \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"model\": \"my-gpt-oss\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
        print(f"\n💡 Note: Model will be downloaded from HuggingFace on first use")
        print(f"   This requires disk space and network access to huggingface.co")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
