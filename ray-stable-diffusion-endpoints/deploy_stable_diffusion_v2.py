#!/usr/bin/env python3
"""
Deploy Stable Diffusion endpoints
Version 2: Separate applications per model with autoscaling and compute reporting
"""

import argparse
import sys
import os
import time
import glob

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

# Import the Stable Diffusion app
try:
    from serve_stable_diffusion_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_stable_diffusion_v2: {e}")
    print("   Make sure serve_stable_diffusion_v2.py is in the same directory")
    sys.exit(1)


def find_models(model_dir: str = "/data/models/stablediffusion"):
    """Find all .safetensors files in the model directory"""
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for file in glob.glob(os.path.join(model_dir, "*.safetensors")):
        model_name = os.path.basename(file).replace(".safetensors", "")
        models.append({
            "name": model_name,
            "path": file
        })
    
    return sorted(models, key=lambda x: x["name"])


def deploy_stable_diffusion(model_name: str, model_path: str, app_name: str = None):
    """Deploy Stable Diffusion endpoint with separate application"""
    
    print(f"🚀 Deploying Stable Diffusion endpoint...")
    print(f"   Model: {model_name}")
    print(f"   Model path: {model_path}")
    
    # Create unique app name from model name
    if app_name is None:
        app_name = f"sd-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    print(f"   Application name: {app_name}")
    
    # Create the application
    app = create_app(model_name=model_name, model_path=model_path)
    
    # Deploy using serve.run with unique app name
    serve.run(
        app,
        name=app_name,  # Separate application per model
        route_prefix=f"/{app_name}/v1",  # Unique route per model
    )
    
    model_id = os.path.basename(model_path).replace(".safetensors", "").replace(".ckpt", "")
    print(f"✅ Stable Diffusion endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   API: http://<head-node-ip>:8000/{app_name}/v1")
    print(f"   Model: {model_id}")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")


def main():
    parser = argparse.ArgumentParser(description="Deploy Stable Diffusion endpoint")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to .safetensors model file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (default: filename without extension)",
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default=None,
        help="Application name (default: auto-generated from model name)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in /data/models/stablediffusion",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("📋 Available Stable Diffusion models:")
        models = find_models()
        if not models:
            print("   No .safetensors files found in /data/models/stablediffusion")
        else:
            for i, model in enumerate(models, 1):
                size_mb = os.path.getsize(model["path"]) / (1024 * 1024)
                print(f"   {i}. {model['name']} ({size_mb:.0f} MB)")
        return
    
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
    
    # Start Ray Serve with HTTP options
    RAY_SERVE_PORT = 8000
    try:
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
        try:
            serve.start(detached=True, http_options=http_options)
            print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
        except Exception as start_error:
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
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not args.model_path.endswith('.safetensors'):
        print(f"⚠️  Warning: Model file is not .safetensors: {args.model_path}")
        print(f"   Only .safetensors files are supported")
    
    # Get model name
    model_name = args.model_name or os.path.basename(args.model_path).replace(".safetensors", "")
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_stable_diffusion(
            model_name=model_name,
            model_path=args.model_path,
            app_name=args.app_name
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"sd-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            
            print("\n📊 Deployment Summary:")
            print(f"   Model: {model_name}")
            print(f"   Model path: {args.model_path}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Deployment time: {deploy_time:.2f}s")
            print(f"\n📝 Example usage:")
            print(f"   curl -X POST http://{head_node_ip}:{RAY_SERVE_PORT}/{app_name}/v1/generate \\")
            print(f"     -H 'Content-Type: application/json' \\")
            print(f"     -d '{{\"prompt\": \"a beautiful landscape\", \"num_inference_steps\": 50}}'")
        except:
            app_name = args.app_name or f"sd-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            print("\n📊 Deployment Summary:")
            print(f"   Model: {model_name}")
            print(f"   Application: {app_name}")
            print(f"   Endpoint: http://<head-node-ip>:{RAY_SERVE_PORT}/{app_name}/v1")
            print(f"   Deployment time: {deploy_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
