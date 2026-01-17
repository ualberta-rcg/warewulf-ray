#!/usr/bin/env python3
"""
Deploy Kandinsky endpoints
Version 2: Separate applications per model with autoscaling and dynamic input discovery
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

# Import the Kandinsky app
try:
    from serve_kandinsky_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_kandinsky_v2: {e}")
    print("   Make sure serve_kandinsky_v2.py is in the same directory")
    sys.exit(1)


def find_models(model_dir: str = "/data/models/stablediffusion"):
    """Find all Kandinsky model files (.ckpt and .safetensors) in the model directory"""
    if not os.path.exists(model_dir):
        return []
    
    models = []
    for pattern in ["*.ckpt", "*.safetensors"]:
        for file in glob.glob(os.path.join(model_dir, pattern)):
            # Filter for Kandinsky models (by filename)
            filename = os.path.basename(file).lower()
            if "kandinsky" in filename:
                model_name = os.path.basename(file).replace(".safetensors", "").replace(".ckpt", "")
                models.append({
                    "name": model_name,
                    "path": file
                })
    
    return sorted(models, key=lambda x: x["name"])


def deploy_kandinsky(model_name: str, model_path: str, app_name: str = None):
    """Deploy Kandinsky endpoint with separate application"""
    
    print(f"🚀 Deploying Kandinsky endpoint...")
    print(f"   Model: {model_name}")
    print(f"   Model path: {model_path}")
    
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
    
    model_id = os.path.basename(model_path).replace(".safetensors", "").replace(".ckpt", "")
    print(f"✅ Kandinsky endpoint deployed successfully!")
    print(f"   Application: {app_name}")
    print(f"   API: http://<head-node-ip>:8000/{app_name}/v1")
    print(f"   Model: {model_id}")
    print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
    print(f"   Schema endpoint: http://<head-node-ip>:8000/{app_name}/v1/schema")


def main():
    parser = argparse.ArgumentParser(description="Deploy Kandinsky endpoint")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to .ckpt or .safetensors model file",
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
        help="List available Kandinsky models in /data/models/stablediffusion",
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
        print("📋 Available Kandinsky models:")
        models = find_models()
        if not models:
            print("   No Kandinsky model files found in /data/models/stablediffusion")
            print("   (Looking for files with 'kandinsky' in the name)")
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
    
    # Ray Serve should already be running on the cluster
    # Just connect and deploy (no need to start/restart Serve)
    
    # Validate model path (if not listing)
    if not args.list_models:
        if not args.model_path:
            print(f"❌ --model-path is required when deploying")
            sys.exit(1)
        
        # Check if it's a HuggingFace model ID (contains '/') or a local path
        is_hf_model_id = '/' in args.model_path and not os.path.isabs(args.model_path) and not os.path.exists(args.model_path)
        is_local_file = os.path.isfile(args.model_path)
        is_local_dir = os.path.isdir(args.model_path)
        
        if not (is_hf_model_id or is_local_file or is_local_dir):
            print(f"❌ Model path not found: {args.model_path}")
            print(f"   Expected: HuggingFace model ID (e.g., 'ai-forever/Kandinsky3.1'),")
            print(f"            local file path, or local directory path")
            sys.exit(1)
        
        if is_local_file and not (args.model_path.endswith('.ckpt') or args.model_path.endswith('.safetensors')):
            print(f"⚠️  Warning: Model file is not .ckpt or .safetensors: {args.model_path}")
        
        if is_hf_model_id:
            print(f"📦 Using HuggingFace model ID: {args.model_path}")
            print(f"   Model will be downloaded automatically on first use")
    
    # Get model name
    model_name = args.model_name or os.path.basename(args.model_path).replace(".ckpt", "").replace(".safetensors", "")
    
    # Deploy the endpoint
    try:
        deploy_start = time.time()
        deploy_kandinsky(
            model_name=model_name,
            model_path=args.model_path,
            app_name=args.app_name
        )
        deploy_time = time.time() - deploy_start
        
        # Get head node IP for user
        try:
            head_node_ip = ray.util.get_node_ip_address()
            app_name = args.app_name or f"kandinsky-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            
            print("\n📊 Deployment Summary:")
            print(f"   Model: {model_name}")
            print(f"   Model path: {args.model_path}")
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
            app_name = args.app_name or f"kandinsky-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
            print("\n📊 Deployment Summary:")
            print(f"   Model: {model_name}")
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
