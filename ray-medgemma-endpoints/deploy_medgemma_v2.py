#!/usr/bin/env python3
"""
Deploy MedGemma endpoints
Version 2: Separate applications per model with autoscaling and dynamic input discovery
"""

import argparse
import sys
import os
import time

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

# Import the MedGemma app
try:
    from serve_medgemma_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_medgemma_v2: {e}")
    print("   Make sure serve_medgemma_v2.py is in the same directory")
    sys.exit(1)


def deploy_medgemma(model_name: str, model_path: str, app_name: str = None, serve_port: int = 9202, hf_token: str = None, num_gpus: int = 1):
    """Deploy MedGemma endpoint with separate application"""
    
    print(f"🚀 Deploying MedGemma endpoint...")
    print(f"   Model ID: {model_path}")
    print(f"   Model name: {model_name}")
    print(f"   GPUs per replica: {num_gpus}")
    
    # Create unique app name from model name
    if app_name is None:
        app_name = f"medgemma-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(model_name=model_name, model_path=model_path, hf_token=hf_token, num_gpus=num_gpus)
    
    # Deploy using serve.run with unique app name
    serve.run(
        app,
        name=app_name,
        route_prefix=f"/{app_name}/v1",
    )
    
    print(f"✅ MedGemma endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   API: http://<head-node-ip>:{serve_port}/{app_name}/v1")
    print(f"   Model ID: {model_path}")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
    print(f"   Schema endpoint: http://<head-node-ip>:{serve_port}/{app_name}/v1/schema")
    print(f"   ⚠️  Note: 27B model is large - first request may take 2-5 minutes to load")


def main():
    parser = argparse.ArgumentParser(description="Deploy MedGemma endpoint")
    parser.add_argument(
        "--app-name",
        type=str,
        default=None,
        help="Application name (default: auto-generated)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="google/medgemma-27b-it",
        help="HuggingFace model path (default: google/medgemma-27b-it)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto - detects head node)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9202,
        help="Ray Serve HTTP port (default: 9202)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per replica (default: 1, recommend 2-4 for 27B model)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force replace existing application (deletes it first)",
    )
    
    args = parser.parse_args()
    
    # Get HF token from argument or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Use MedGemma 27B from HuggingFace
    model_id = args.model_path
    
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
    
    # Check for transformers
    print("🔍 Checking dependencies...")
    try:
        import transformers
        print(f"✅ Transformers available (version: {transformers.__version__})")
        if transformers.__version__ < "4.50.0":
            print("   ⚠️  Warning: Transformers version < 4.50.0 may not support MedGemma")
            print("   Transformers will be installed automatically on worker nodes via runtime_env")
    except ImportError:
        print("⚠️  Transformers not found.")
        print("   Note: Transformers will be installed automatically on worker nodes via runtime_env")
    
    # Check for HF token (may be required for gated models)
    if hf_token:
        print("✅ HuggingFace token found (required for gated models)")
        print(f"   Token length: {len(hf_token)} characters")
    else:
        print("⚠️  No HuggingFace token found - may be required for MedGemma (gated model)")
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
            print(f"   MedGemma will be deployed alongside existing deployments")
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
    
    # Use MedGemma 27B from HuggingFace
    print(f"📦 Using MedGemma from HuggingFace: {model_id}")
    print(f"   Model will be downloaded automatically on first use")
    print(f"   ⚠️  Note: 27B model is large (~50GB+) and may take 10-30 minutes to download")
    
    # Get model name from app name or use default
    model_name = args.app_name or "medgemma-27b-it"
    
    # Force replace if requested
    if args.force:
        app_name = args.app_name or f"medgemma-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
        print(f"🔄 Force replace requested for: {app_name}")
        try:
            status = serve.status()
            if hasattr(status, 'applications') and app_name in status.applications:
                print(f"   Deleting existing application: {app_name}")
                serve.delete(app_name)
                print(f"   ✅ Deleted existing application")
                time.sleep(1)  # Give it a moment to fully delete
            else:
                print(f"   Application '{app_name}' not found (will create new)")
        except Exception as e:
            print(f"   ⚠️  Could not delete existing application: {e}")
            print(f"   Continuing with deployment anyway...")
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_medgemma(
            model_name=model_name,
            model_path=model_id,
            app_name=args.app_name,
            serve_port=RAY_SERVE_PORT,
            hf_token=hf_token,
            num_gpus=args.num_gpus
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"medgemma-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_id}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Get schema (discover inputs):")
            print(f"   curl http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   # Text-only query:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"text\": \"How do you differentiate bacterial from viral pneumonia?\", \\")
            print(f"           \"system\": \"You are a helpful medical assistant.\", \\")
            print(f"           \"max_new_tokens\": 200}}'")
            print(f"   # Image + text query (X-ray analysis):")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"text\": \"Describe this X-ray\", \\")
            print(f"           \"image\": \"data:image/png;base64,...\", \\")
            print(f"           \"system\": \"You are an expert radiologist.\", \\")
            print(f"           \"max_new_tokens\": 300}}'")
            print(f"   # Using chat format (messages):")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"messages\": [\\")
            print(f"         {{\"role\": \"system\", \"content\": [{{\"type\": \"text\", \"text\": \"You are a helpful medical assistant.\"}}]}}, \\")
            print(f"         {{\"role\": \"user\", \"content\": [{{\"type\": \"text\", \"text\": \"What are the symptoms of diabetes?\"}}]}}\\")
            print(f"       ]}}'")
        except:
            app_name = args.app_name or model_name
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_id}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Get schema (discover inputs):")
            print(f"   curl http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   # Text-only query:")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"text\": \"How do you differentiate bacterial from viral pneumonia?\", \\")
            print(f"           \"system\": \"You are a helpful medical assistant.\", \\")
            print(f"           \"max_new_tokens\": 200}}'")
            print(f"   # Image + text query (X-ray analysis):")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"text\": \"Describe this X-ray\", \\")
            print(f"           \"image\": \"data:image/png;base64,...\", \\")
            print(f"           \"system\": \"You are an expert radiologist.\", \\")
            print(f"           \"max_new_tokens\": 300}}'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
