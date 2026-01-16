#!/usr/bin/env python3
"""
Deploy multiple HuggingFace LLM models with different route prefixes
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

# Import the vLLM-based HuggingFace app
try:
    from serve_hf_models import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_hf_models: {e}")
    print("   Make sure serve_hf_models.py is in the same directory")
    sys.exit(1)


def deploy_multiple_models(models_config):
    """Deploy multiple HuggingFace LLM models
    
    Args:
        models_config: List of dicts with 'model_name', 'route_prefix', and optionally 'hf_token'
    """
    
    print(f"🚀 Deploying {len(models_config)} HuggingFace LLM models...")
    
    apps = {}
    for i, config in enumerate(models_config):
        model_name = config['model_name']
        route_prefix = config.get('route_prefix', f"/v{i+1}")
        hf_token = config.get('hf_token')
        
        print(f"\n   Model {i+1}: {model_name}")
        print(f"   Route: {route_prefix}")
        
        # Create the application
        app = create_app(model_name=model_name, hf_token=hf_token)
        apps[route_prefix] = app
    
    # Deploy all applications
    serve.run(apps)
    
    print(f"\n✅ All models deployed successfully!")
    for i, config in enumerate(models_config):
        route_prefix = config.get('route_prefix', f"/v{i+1}")
        print(f"   {config['model_name']}: http://<head-node-ip>:8000{route_prefix}")


def main():
    parser = argparse.ArgumentParser(description="Deploy multiple HuggingFace LLM models")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., 'model1,model2')",
    )
    parser.add_argument(
        "--routes",
        type=str,
        default=None,
        help="Comma-separated list of route prefixes (e.g., '/v1,/v2'). Default: auto-assigned",
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
    
    # Parse models and routes
    model_names = [m.strip() for m in args.models.split(",")]
    if args.routes:
        routes = [r.strip() for r in args.routes.split(",")]
        if len(routes) != len(model_names):
            print("❌ Number of routes must match number of models")
            sys.exit(1)
    else:
        routes = [f"/v{i+1}" for i in range(len(model_names))]
    
    # Build models config
    models_config = [
        {
            'model_name': model_name,
            'route_prefix': route,
            'hf_token': hf_token
        }
        for model_name, route in zip(model_names, routes)
    ]
    
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
    RAY_SERVE_PORT = 8000
    try:
        # Check if Serve is already running
        try:
            status = serve.status()
            print(f"✅ Ray Serve is already running")
            print(f"   Current applications: {len(status.applications) if hasattr(status, 'applications') else 'N/A'}")
            # Restart Serve to ensure it's listening on 0.0.0.0
            print(f"🔄 Restarting Ray Serve to ensure it's listening on 0.0.0.0:{RAY_SERVE_PORT}...")
            serve.shutdown()
            import time
            time.sleep(1)  # Give it time to shutdown
        except:
            pass  # Serve not running, will start below
        
        # Start Serve with correct options
        http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
        serve.start(detached=True, http_options=http_options)
        print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT}")
    except Exception as e:
        print(f"⚠️  Ray Serve setup issue: {e}")
        print(f"   Continuing anyway...")
    
    # Deploy the models
    try:
        deploy_multiple_models(models_config)
        
        print("\n📊 Deployment Summary:")
        for i, config in enumerate(models_config):
            route_prefix = config.get('route_prefix', f"/v{i+1}")
            print(f"   Model {i+1}: {config['model_name']}")
            print(f"   Route: http://<head-node-ip>:{RAY_SERVE_PORT}{route_prefix}")
            print(f"   Example:")
            model_id = config['model_name'].replace('/', '-')
            print(f"     curl http://<head-node-ip>:{RAY_SERVE_PORT}{route_prefix}/chat/completions \\")
            print(f"       -H 'Content-Type: application/json' \\")
            print(f"       -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
            print()
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
