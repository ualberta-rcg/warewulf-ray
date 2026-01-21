#!/usr/bin/env python3
"""
Deploy Kandinsky endpoints
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

# Import the Kandinsky app
try:
    from serve_kandinsky_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_kandinsky_v2: {e}")
    print("   Make sure serve_kandinsky_v2.py is in the same directory")
    sys.exit(1)


# Removed find_models function - no longer needed since we only use HuggingFace models


def deploy_kandinsky(model_name: str, model_path: str, app_name: str = None):
    """Deploy Kandinsky endpoint with separate application"""
    
    print(f"🚀 Deploying Kandinsky endpoint...")
    print(f"   Model ID: {model_path}")
    print(f"   Model name: {model_name}")
    
    # Create unique app name from model name
    if app_name is None:
        app_name = f"kandinsky-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(model_name=model_name, model_path=model_path)
    
    # Deploy using serve.run with unique app name
    serve.run(
        app,
        name=app_name,
        route_prefix=f"/{app_name}/v1",
    )
    
    print(f"✅ Kandinsky endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   API: http://<head-node-ip>:8000/{app_name}/v1")
    print(f"   Model ID: {model_path}")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
    print(f"   Schema endpoint: http://<head-node-ip>:8000/{app_name}/v1/schema")


def main():
    parser = argparse.ArgumentParser(description="Deploy Kandinsky endpoint")
    parser.add_argument(
        "--app-name",
        type=str,
        default=None,
        help="Application name (default: auto-generated)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Always use Kandinsky 3.1 from HuggingFace
    model_id = "ai-forever/Kandinsky3.1"
    
    # Initialize Ray connection
    try:
        ray.init(address=args.address, ignore_reinit_error=True)
        print(f"✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        sys.exit(1)
    
    # Check for diffusers
    print("🔍 Checking dependencies...")
    try:
        import diffusers
        print(f"✅ Diffusers available (version: {diffusers.__version__})")
    except ImportError:
        print("⚠️  Diffusers not found.")
        print("   Note: Diffusers will be installed automatically on worker nodes via runtime_env")
    
    # Start Ray Serve with HTTP options to listen on all interfaces
    RAY_SERVE_PORT = 8000  # Default Ray Serve port
    try:
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
        try:
            serve.start(detached=True, http_options=http_options)
            print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
        except Exception as start_error:
            # Serve might already be running - check and restart if needed
            try:
                status = serve.status()
                print(f"🔄 Ray Serve is already running, restarting to apply HTTP options...")
                serve.shutdown()
                time.sleep(2)
                serve.start(detached=True, http_options=http_options)
                print(f"✅ Ray Serve restarted on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
            except Exception as restart_error:
                print(f"⚠️  Ray Serve setup issue: {restart_error}")
                print(f"   Continuing anyway - Serve may already be configured correctly")
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Continuing anyway...")
    
    # Always use Kandinsky 3.1
    print(f"📦 Using Kandinsky 3.1 from HuggingFace: {model_id}")
    print(f"   Model will be downloaded automatically on first use")
    
    # Get model name from app name or use default
    model_name = args.app_name or "kandinsky3-1"
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_kandinsky(
            model_name=model_name,
            model_path=model_id,  # Pass model ID as model_path (it will be treated as HuggingFace ID)
            app_name=args.app_name
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"kandinsky-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_id}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Get schema (discover inputs):")
            print(f"   curl http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   # Generate image:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a beautiful landscape\", \"num_inference_steps\": 50}}'")
        except:
            app_name = args.app_name or model_name
            print("\n📊 Deployment Summary:")
            print(f"   Model ID: {model_id}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Schema: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/schema")
            print(f"   Deployment time: {deploy_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
