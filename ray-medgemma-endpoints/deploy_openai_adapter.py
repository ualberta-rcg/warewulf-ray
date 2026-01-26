#!/usr/bin/env python3
"""
Deploy OpenAI API Adapter for MedGemma
This adapter proxies OpenAI Chat Completions format requests to the existing MedGemma endpoint
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

# Import the adapter app
try:
    from serve_openai_adapter import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_openai_adapter: {e}")
    print("   Make sure serve_openai_adapter.py is in the same directory")
    sys.exit(1)


def deploy_adapter(medgemma_base_url: str, model_name: str, app_name: str = "openai-adapter", serve_port: int = 9202):
    """Deploy OpenAI adapter endpoint"""
    
    print(f"🚀 Deploying OpenAI adapter...")
    print(f"   MedGemma endpoint: {medgemma_base_url}")
    print(f"   Model name: {model_name}")
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(medgemma_base_url=medgemma_base_url, model_name=model_name)
    
    # Deploy using serve.run
    serve.run(
        app,
        name=app_name,
        route_prefix=f"/{app_name}",
    )
    
    print(f"✅ OpenAI adapter deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   OpenAI API: http://<head-node-ip>:{serve_port}/{app_name}/v1/chat/completions")
    print(f"   OpenAI Models: http://<head-node-ip>:{serve_port}/{app_name}/v1/models")
    print(f"   Proxies to: {medgemma_base_url}")


def main():
    parser = argparse.ArgumentParser(description="Deploy OpenAI adapter for MedGemma")
    parser.add_argument(
        "--medgemma-url",
        type=str,
        default="http://172.26.92.232:9202/medgemma-27b-it/v1",
        help="Base URL of the MedGemma endpoint (default: http://172.26.92.232:9202/medgemma-27b-it/v1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="medgemma-27b-it",
        help="Model name for OpenAI API (default: medgemma-27b-it)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default="openai-adapter",
        help="Application name (default: openai-adapter)",
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
        "--force",
        action="store_true",
        help="Force replace existing application (deletes it first)",
    )
    
    args = parser.parse_args()
    
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
    
    # Start Ray Serve with HTTP options to listen on all interfaces (0.0.0.0)
    RAY_SERVE_PORT = args.port
    RAY_SERVE_HOST = "0.0.0.0"  # Listen on all interfaces
    
    try:
        from ray.serve.config import HTTPOptions
        
        # Check if Ray Serve is already running
        try:
            status = serve.status()
            print(f"✅ Ray Serve is already running")
            if hasattr(status, 'applications') and status.applications:
                app_count = len(status.applications)
                print(f"   Found {app_count} existing application(s) - will not affect them")
        except:
            # Serve not running, start it
            print(f"🚀 Starting Ray Serve on {RAY_SERVE_HOST}:{RAY_SERVE_PORT}...")
            http_options = HTTPOptions(host=RAY_SERVE_HOST, port=RAY_SERVE_PORT)
            serve.start(detached=True, http_options=http_options)
            print(f"✅ Ray Serve started on {RAY_SERVE_HOST}:{RAY_SERVE_PORT}")
            
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Attempting to continue anyway...")
        import traceback
        traceback.print_exc()
    
    # Force replace if requested
    if args.force:
        print(f"🔄 Force replace requested for: {args.app_name}")
        try:
            status = serve.status()
            if hasattr(status, 'applications') and args.app_name in status.applications:
                print(f"   Deleting existing application: {args.app_name}")
                serve.delete(args.app_name)
                print(f"   ✅ Deleted existing application")
                time.sleep(1)  # Give it a moment to fully delete
            else:
                print(f"   Application '{args.app_name}' not found (will create new)")
        except Exception as e:
            print(f"   ⚠️  Could not delete existing application: {e}")
            print(f"   Continuing with deployment anyway...")
    
    # Deploy the adapter
    try:
        deploy_start = time.time()
        deploy_adapter(
            medgemma_base_url=args.medgemma_url,
            model_name=args.model_name,
            app_name=args.app_name,
            serve_port=RAY_SERVE_PORT
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            
            print("\n📊 Deployment Summary:")
            print(f"   Application: {args.app_name}")
            print(f"   OpenAI Chat Completions: http://{head_node_ip}:{RAY_SERVE_PORT}/{args.app_name}/v1/chat/completions")
            print(f"   OpenAI Models List: http://{head_node_ip}:{RAY_SERVE_PORT}/{args.app_name}/v1/models")
            print(f"   Model Name: {args.model_name}")
            print(f"   Proxies to: {args.medgemma_url}")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   # Chat completion:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{args.app_name}/v1/chat/completions \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -H 'Authorization: Bearer sk-xxx' \\")
            print(f"     -d '{{")
            print(f"       \"model\": \"{args.model_name}\",")
            print(f"       \"messages\": [")
            print(f"         {{\"role\": \"user\", \"content\": \"What are the symptoms of diabetes?\"}}")
            print(f"       ],")
            print(f"       \"temperature\": 0.7,")
            print(f"       \"max_tokens\": 200")
            print(f"     }}'")
            print(f"\n   # For OpenWebUI:")
            print(f"   Base URL: http://{head_node_ip}:{RAY_SERVE_PORT}/{args.app_name}")
            print(f"   API Key: sk-any (OpenWebUI will work with any key)")
        except:
            print("\n📊 Deployment Summary:")
            print(f"   Application: {args.app_name}")
            print(f"   OpenAI Chat Completions: http://<head-node-ip>:{RAY_SERVE_PORT}/{args.app_name}/v1/chat/completions")
            print(f"   OpenAI Models List: http://<head-node-ip>:{RAY_SERVE_PORT}/{args.app_name}/v1/models")
            print(f"   Model Name: {args.model_name}")
            print(f"   Proxies to: {args.medgemma_url}")
            print(f"   Deployment time: {deploy_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
