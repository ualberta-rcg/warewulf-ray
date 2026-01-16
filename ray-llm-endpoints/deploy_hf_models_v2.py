#!/usr/bin/env python3
"""
Deploy HuggingFace LLM endpoints using vLLM directly
Version 2: Separate applications per model with autoscaling and compute reporting
"""

import argparse
import sys
import os
import time
import json

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

# Import the vLLM-based HuggingFace app
try:
    from serve_hf_models_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_hf_models_v2: {e}")
    print("   Make sure serve_hf_models_v2.py is in the same directory")
    sys.exit(1)


# Model configurations with context lengths
MODEL_CONFIGS = {
    "deepseek-ai/deepseek-llm-7b-chat": {
        "max_model_len": 4096,  # DeepSeek-7B supports 4096 tokens
        "description": "DeepSeek-7B Chat model"
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "max_model_len": 8192,  # Mistral-7B supports 8192 tokens
        "description": "Mistral-7B Instruct model (larger context)"
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "max_model_len": 32768,  # Mistral-7B v0.2 supports 32k tokens
        "description": "Mistral-7B Instruct v0.2 (very large context)"
    },
}


def get_model_config(model_name: str):
    """Get configuration for a model"""
    return MODEL_CONFIGS.get(model_name, {
        "max_model_len": 4096,  # Default
        "description": "Custom model"
    })


def deploy_hf_llm(model_name: str, hf_token: str = None, app_name: str = None):
    """Deploy HuggingFace LLM endpoint using vLLM with separate application"""
    
    config = get_model_config(model_name)
    max_model_len = config["max_model_len"]
    
    print(f"🚀 Deploying HuggingFace LLM endpoint (vLLM direct)...")
    print(f"   Model: {model_name}")
    print(f"   Max context length: {max_model_len} tokens")
    print(f"   Description: {config['description']}")
    if hf_token:
        print(f"   Using HuggingFace token (length: {len(hf_token)})")
    
    # Create unique app name from model name
    if app_name is None:
        app_name = f"hf-{model_name.replace('/', '-').replace('_', '-')}"
    
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(model_name=model_name, hf_token=hf_token, max_model_len=max_model_len)
    
    # Deploy using serve.run with unique app name
    serve.run(
        app,
        name=app_name,  # Separate application per model
        route_prefix=f"/{app_name}/v1",  # Unique route per model
    )
    
    model_id = model_name.replace("/", "-")
    print(f"✅ HuggingFace LLM endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   OpenAI API: http://<head-node-ip>:8000/{app_name}/v1")
    print(f"   Model name: {model_id}")
    print(f"   Max context: {max_model_len} tokens")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")


def get_compute_report():
    """Generate compute usage report"""
    try:
        import ray
        
        # Get cluster resources
        resources = ray.cluster_resources()
        available = ray.available_resources()
        
        # Get node info
        nodes = ray.nodes()
        
        report = {
            "timestamp": time.time(),
            "cluster_resources": {
                "total_cpus": resources.get("CPU", 0),
                "total_gpus": resources.get("GPU", 0),
                "total_memory_gb": resources.get("memory", 0) / (1024**3),
            },
            "available_resources": {
                "available_cpus": available.get("CPU", 0),
                "available_gpus": available.get("GPU", 0),
                "available_memory_gb": available.get("memory", 0) / (1024**3),
            },
            "nodes": len(nodes),
            "gpu_nodes": sum(1 for node in nodes if node.get("Resources", {}).get("GPU", 0) > 0),
        }
        
        # Get Serve status
        try:
            from ray import serve
            status = serve.status()
            report["serve"] = {
                "applications": len(status.applications) if hasattr(status, 'applications') else 0,
            }
        except:
            report["serve"] = {"error": "Could not get Serve status"}
        
        return report
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Deploy HuggingFace LLM endpoint (vLLM direct)")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-chat",
        help="Model to deploy (HuggingFace model ID)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for private models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default=None,
        help="Application name (default: auto-generated from model name)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    parser.add_argument(
        "--compute-report",
        action="store_true",
        help="Generate compute usage report after deployment",
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
        deploy_start = time.time()
        deploy_hf_llm(model_name=args.model, hf_token=hf_token, app_name=args.app_name)
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"hf-{args.model.replace('/', '-').replace('_', '-')}"
            model_id = args.model.replace("/", "-")
            config = get_model_config(args.model)
            
            print("\n📊 Deployment Summary:")
            print(f"   Model: {args.model}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Model name: {model_id}")
            print(f"   Max context: {config['max_model_len']} tokens")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   curl http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
        except:
            app_name = args.app_name or f"hf-{args.model.replace('/', '-').replace('_', '-')}"
            model_id = args.model.replace("/", "-")
            config = get_model_config(args.model)
            print("\n📊 Deployment Summary:")
            print(f"   Model: {args.model}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Model name: {model_id}")
            print(f"   Max context: {config['max_model_len']} tokens")
            print(f"   Deployment time: {deploy_time:.2f}s")
        
        print(f"\n💡 Note: Model will be downloaded from HuggingFace on first use")
        print(f"   This requires disk space and network access to huggingface.co")
        if hf_token:
            print(f"   Using HuggingFace token for authentication")
        
        # Generate compute report if requested
        if args.compute_report:
            print("\n📈 Generating compute usage report...")
            report = get_compute_report()
            print(json.dumps(report, indent=2))
            print("\n💾 Report saved to compute_report.json")
            with open("compute_report.json", "w") as f:
                json.dump(report, f, indent=2)
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
