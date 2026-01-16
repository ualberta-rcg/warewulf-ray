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
    # Note: If deploying multiple models, they can share /v1 route with different model names
    serve.run(
        app,
        route_prefix="/v1",  # OpenAI-compatible API endpoint (shared with other models)
    )
    
    model_id = model_name.replace("/", "-")
    print(f"✅ LLM endpoint deployed successfully!")
    print(f"   OpenAI API: http://<head-node-ip>:8000/v1")
    print(f"   Model name: {model_id}")


def main():
    parser = argparse.ArgumentParser(description="Deploy GPT-OSS LLM endpoint (vLLM direct)")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model to deploy (HuggingFace model ID, default: meta-llama/Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for private models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Get HF token from arg or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
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
    # Always restart to ensure HTTP options are applied (like ray-endpoints does)
    RAY_SERVE_PORT = 8000  # Use 8000 (user prefers this port)
    try:
        from ray.serve.config import HTTPOptions
        # Always try to start with HTTP options (will fail if already running, that's OK)
        http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
        try:
            serve.start(detached=True, http_options=http_options)
            print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
        except Exception as start_error:
            # Serve might already be running - restart it to apply HTTP options
            try:
                status = serve.status()
                print(f"🔄 Ray Serve is already running, restarting to apply HTTP options...")
                serve.shutdown()
                import time
                time.sleep(2)  # Give it time to fully shutdown
                serve.start(detached=True, http_options=http_options)
                print(f"✅ Ray Serve restarted on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
            except Exception as restart_error:
                print(f"⚠️  Ray Serve setup issue: {restart_error}")
                print(f"   Continuing anyway - Serve may already be configured correctly")
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Continuing anyway...")
    
    # Deploy the endpoint
    try:
        deploy_gpt_oss(model_name=args.model, hf_token=hf_token)
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            model_id = args.model.replace("/", "-")
            print("\n📊 Deployment Summary:")
            print(f"   Model: {args.model}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/v1")
            print(f"   Model name: {model_id}")
            print(f"\n📝 Example usage:")
            print(f"   curl http://{head_node_ip}:{RAY_SERVE_PORT}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
        except:
            model_id = args.model.replace("/", "-")
            print("\n📊 Deployment Summary:")
            print(f"   Model: {args.model}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/v1")
            print(f"   Model name: {model_id}")
            print(f"\n📝 Example usage:")
            print(f"   curl http://<head-node-ip>:{RAY_SERVE_PORT}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
        
        print(f"\n💡 Note: Model will be downloaded from HuggingFace on first use")
        print(f"   This requires disk space and network access to huggingface.co")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
