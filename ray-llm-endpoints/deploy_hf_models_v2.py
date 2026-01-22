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


def deploy_hf_llm(model_name: str, model_path: str = None, app_name: str = None, serve_port: int = 9200, hf_token: str = None, max_model_len: int = None):
    """Deploy HuggingFace LLM endpoint using vLLM with separate application"""
    
    if model_path is None:
        model_path = model_name
    
    config = get_model_config(model_name)
    if max_model_len is None:
        max_model_len = config["max_model_len"]
    
    print(f"🚀 Deploying HuggingFace LLM endpoint...")
    print(f"   Model ID: {model_path}")
    print(f"   Model name: {model_name}")
    print(f"   Max context length: {max_model_len} tokens")
    print(f"   Description: {config['description']}")
    if hf_token:
        print(f"   Using HuggingFace token (length: {len(hf_token)})")
    
    # Create unique app name from model name
    if app_name is None:
        app_name = f"hf-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(model_name=model_name, model_path=model_path, hf_token=hf_token, max_model_len=max_model_len)
    
    # Deploy using serve.run with unique app name
    serve.run(
        app,
        name=app_name,
        route_prefix=f"/{app_name}/v1",
    )
    
    print(f"✅ HuggingFace LLM endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   API: http://<head-node-ip>:{serve_port}/{app_name}/v1")
    print(f"   Model ID: {model_path}")
    print(f"   Max context: {max_model_len} tokens")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
    print(f"   Schema endpoint: http://<head-node-ip>:{serve_port}/{app_name}/v1/schema")


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
    
    parser.add_argument(
        "--port",
        type=int,
        default=9200,
        help="Ray Serve HTTP port (default: 9200)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (default: auto-detected from model config)",
    )
    
    args = parser.parse_args()
    
    # Get HF token from argument or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Use model name as model path if not specified
    model_id = args.model
    model_path = model_id
    
    # Initialize Ray connection - auto-detect head node IP
    ray_address = args.address
    if ray_address == "auto":
        # Auto-detect head node IP and port
        print("🔍 Auto-detecting Ray head node...")
        try:
            # Try to connect to auto first to get cluster info
            ray.init(address="auto", ignore_reinit_error=True)
            
            # Get the head node IP from the cluster
            nodes = ray.nodes()
            head_node = None
            for node in nodes:
                # Head node has this resource
                if node.get('Resources', {}).get('node:__internal_head__', 0) == 1.0:
                    head_node = node
                    break
            
            if head_node:
                head_node_ip = head_node.get('NodeManagerAddress', '127.0.0.1')
                ray_port = 6379  # Default Ray port
                ray_address = f"{head_node_ip}:{ray_port}"
                print(f"✅ Detected head node at {ray_address}")
                # Shutdown the auto connection
                ray.shutdown()
            else:
                # Fallback: use current node IP
                import socket
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                ray_address = f"{local_ip}:6379"
                print(f"⚠️  Could not detect head node, using local IP: {ray_address}")
        except Exception as detect_err:
            # Fallback: use localhost or detect from environment
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            ray_address = f"{local_ip}:6379"
            print(f"⚠️  Auto-detection failed ({detect_err}), using local IP: {ray_address}")
    
    # Now connect to the detected/explicit address
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"✅ Connected to Ray cluster at {ray_address}")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster at {ray_address}: {e}")
        print(f"   Make sure Ray is running and the address is correct")
        sys.exit(1)
    
    # Check for vLLM
    print("🔍 Checking dependencies...")
    try:
        import vllm
        print(f"✅ vLLM available (version: {vllm.__version__})")
    except ImportError:
        print("⚠️  vLLM not found.")
        print("   Note: vLLM will be installed automatically on worker nodes via runtime_env")
    
    # Check for HF token (may be required for gated models)
    if hf_token:
        print("✅ HuggingFace token found (required for gated models)")
        print(f"   Token length: {len(hf_token)} characters")
    else:
        print("⚠️  No HuggingFace token found - may be required for gated models")
        print("   Set HF_TOKEN environment variable or use --hf-token argument")
    
    # Start Ray Serve with HTTP options to listen on all interfaces (0.0.0.0)
    # IMPORTANT: We only start Ray Serve if it's not running, and we never shutdown
    # existing deployments. Each app is deployed independently.
    RAY_SERVE_PORT = args.port
    RAY_SERVE_HOST = "0.0.0.0"  # Listen on all interfaces
    
    try:
        from ray.serve.config import HTTPOptions
        
        # Check if Ray Serve is already running
        try:
            status = serve.status()
            print(f"✅ Ray Serve is already running")
            # Check if there are other applications
            if hasattr(status, 'applications') and status.applications:
                app_count = len(status.applications)
                print(f"   Found {app_count} existing application(s) - will not affect them")
            print(f"   HuggingFace LLM will be deployed alongside existing deployments")
        except:
            # Serve not running, start it
            print(f"🚀 Starting Ray Serve on {RAY_SERVE_HOST}:{RAY_SERVE_PORT}...")
            http_options = HTTPOptions(host=RAY_SERVE_HOST, port=RAY_SERVE_PORT)
            try:
                serve.start(detached=True, http_options=http_options)
                print(f"✅ Ray Serve started on {RAY_SERVE_HOST}:{RAY_SERVE_PORT} (accessible from network)")
            except Exception as start_err:
                # Might already be starting or there's a port conflict
                print(f"⚠️  Could not start Ray Serve: {start_err}")
                print(f"   Attempting to continue - Ray Serve may already be running")
                # Try to get status to confirm
                try:
                    time.sleep(1)
                    status = serve.status()
                    print(f"✅ Ray Serve is running")
                except:
                    print(f"⚠️  Ray Serve status unclear - deployment may fail")
            
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Attempting to continue anyway...")
        import traceback
        traceback.print_exc()
    
    # Get model config
    config = get_model_config(args.model)
    max_model_len = args.max_model_len or config["max_model_len"]
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_hf_llm(
            model_name=args.model,
            model_path=model_path,
            app_name=args.app_name,
            serve_port=RAY_SERVE_PORT,
            hf_token=hf_token,
            max_model_len=max_model_len
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"hf-{args.model.replace('/', '-').replace('_', '-').replace('.', '-')}"
            model_id = args.model.replace("/", "-")
            
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_path}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Max context: {max_model_len} tokens")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Get schema (discover inputs):")
            print(f"   curl http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   # Chat completion:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"{model_id}\", \\")
            print(f"           \"messages\": [{{\"role\": \"user\", \"content\": \"Hello! What is 2+2?\"}}], \\")
            print(f"           \"max_tokens\": 100, \\")
            print(f"           \"temperature\": 0.7}}'")
        except:
            app_name = args.app_name or f"hf-{args.model.replace('/', '-').replace('_', '-').replace('.', '-')}"
            model_id = args.model.replace("/", "-")
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_path}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Max context: {max_model_len} tokens")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Get schema (discover inputs):")
            print(f"   curl http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   # Chat completion:")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"model\": \"{model_id}\", \\")
            print(f"           \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
        
        print(f"\n💡 Note: Model will be downloaded from HuggingFace on first use")
        print(f"   This requires disk space and network access to huggingface.co")
        if hf_token:
            print(f"   Using HuggingFace token for authentication")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
