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


def deploy_kandinsky(model_name: str, model_path: str, app_name: str = None, serve_port: int = 9201):
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
    print(f"   API: http://<head-node-ip>:{serve_port}/{app_name}/v1")
    print(f"   Model ID: {model_path}")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
    print(f"   Schema endpoint: http://<head-node-ip>:{serve_port}/{app_name}/v1/schema")


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
        help="Ray cluster address (default: auto - detects head node)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9201,
        help="Ray Serve HTTP port (default: 9201)",
    )
    
    args = parser.parse_args()
    
    # Use Kandinsky 3 from HuggingFace (community version with proper diffusers structure)
    model_id = "kandinsky-community/kandinsky-3"
    
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
    
    # Check for diffusers
    print("🔍 Checking dependencies...")
    try:
        import diffusers
        print(f"✅ Diffusers available (version: {diffusers.__version__})")
    except ImportError:
        print("⚠️  Diffusers not found.")
        print("   Note: Diffusers will be installed automatically on worker nodes via runtime_env")
    
    # Start Ray Serve with HTTP options to listen on all interfaces (0.0.0.0)
    # This ensures it's accessible from other machines, not just localhost
    RAY_SERVE_PORT = args.port  # Use port from command line argument
    RAY_SERVE_HOST = "0.0.0.0"  # Listen on all interfaces
    
    try:
        from ray.serve.config import HTTPOptions
        
        # Always shutdown and restart to ensure correct host configuration
        # This is important because Ray Serve might have been started with 127.0.0.1
        try:
            status = serve.status()
            print(f"🔄 Ray Serve is already running, shutting down to reconfigure...")
            serve.shutdown()
            time.sleep(3)  # Give it more time to fully shutdown
        except:
            # Serve not running, that's fine
            pass
        
        # Start with correct host (0.0.0.0 to listen on all interfaces)
        http_options = HTTPOptions(host=RAY_SERVE_HOST, port=RAY_SERVE_PORT)
        serve.start(detached=True, http_options=http_options)
        print(f"✅ Ray Serve started on {RAY_SERVE_HOST}:{RAY_SERVE_PORT} (accessible from network)")
        
        # Verify it's actually listening on the right host
        try:
            status = serve.status()
            print(f"   Ray Serve status: {status}")
            # Try to get the actual proxy address
            if hasattr(status, 'proxies') and status.proxies:
                for proxy_id, proxy_info in status.proxies.items():
                    if hasattr(proxy_info, 'node_id'):
                        print(f"   Proxy running on node: {proxy_info.node_id}")
        except Exception as status_err:
            print(f"   Could not get detailed status: {status_err}")
            
    except Exception as e:
        print(f"❌ Ray Serve setup failed: {e}")
        print(f"   Attempting to continue anyway...")
        import traceback
        traceback.print_exc()
    
    # Use Kandinsky 3 (community version with proper diffusers structure)
    print(f"📦 Using Kandinsky 3 from HuggingFace: {model_id}")
    print(f"   Model will be downloaded automatically on first use")
    
    # Get model name from app name or use default
    model_name = args.app_name or "kandinsky3-1"
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_kandinsky(
            model_name=model_name,
            model_path=model_id,  # Pass model ID as model_path (it will be treated as HuggingFace ID)
            app_name=args.app_name,
            serve_port=RAY_SERVE_PORT
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
            print(f"   # Basic text-to-image generation:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a beautiful landscape\", \"num_inference_steps\": 50}}'")
            print(f"   # Advanced generation with all parameters:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a futuristic city at sunset, cyberpunk style\", \\")
            print(f"           \"negative_prompt\": \"blurry, low quality\", \\")
            print(f"           \"num_inference_steps\": 75, \\")
            print(f"           \"guidance_scale\": 7.5, \\")
            print(f"           \"width\": 1024, \\")
            print(f"           \"height\": 1024, \\")
            print(f"           \"seed\": 42}}'")
            print(f"   # Image-to-image transformation:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/image-to-image \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"make it more vibrant and colorful\", \\")
            print(f"           \"image\": \"data:image/png;base64,...\", \\")
            print(f"           \"strength\": 0.75, \\")
            print(f"           \"num_inference_steps\": 50}}'")
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
            print(f"   # Basic text-to-image generation:")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a beautiful landscape\", \"num_inference_steps\": 50}}'")
            print(f"   # Advanced generation with all parameters:")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a futuristic city at sunset, cyberpunk style\", \\")
            print(f"           \"negative_prompt\": \"blurry, low quality\", \\")
            print(f"           \"num_inference_steps\": 75, \\")
            print(f"           \"guidance_scale\": 7.5, \\")
            print(f"           \"width\": 1024, \\")
            print(f"           \"height\": 1024, \\")
            print(f"           \"seed\": 42}}'")
            print(f"   # Image-to-image transformation:")
            print(f"   curl -X POST http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1/image-to-image \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"make it more vibrant and colorful\", \\")
            print(f"           \"image\": \"data:image/png;base64,...\", \\")
            print(f"           \"strength\": 0.75, \\")
            print(f"           \"num_inference_steps\": 50}}'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
