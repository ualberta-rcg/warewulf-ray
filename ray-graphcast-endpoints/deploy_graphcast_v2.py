#!/usr/bin/env python3
"""
Deploy GraphCast weather forecasting endpoints
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
    import ray
    from ray.util import get_node_ip_address
except ImportError:
    print("❌ Ray not found. Please use /opt/ray/bin/python to run this script.")
    print("   Current Python: " + sys.executable)
    sys.exit(1)

# Import the GraphCast app
try:
    from serve_graphcast_v2 import create_app
except ImportError as e:
    print(f"❌ Failed to import serve_graphcast_v2: {e}")
    print("   Make sure serve_graphcast_v2.py is in the same directory")
    sys.exit(1)


def deploy_graphcast(model_name: str, model_path: str, app_name: str = None):
    """Deploy a GraphCast weather forecasting model"""
    
    if app_name is None:
        app_name = f"graphcast-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    print(f"🚀 Deploying GraphCast weather forecasting endpoint...")
    print(f"   Model: {model_name}")
    print(f"   Model path: {model_path}")
    print(f"   Application: {app_name}")
    
    # Initialize Ray (connect to existing cluster)
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except Exception as e:
        print(f"⚠️  Ray init warning: {e}")
    
    # Create and deploy the app (Ray Serve should already be running on cluster)
    app = create_app(model_name, model_path)
    
    try:
        serve.run(app, name=app_name, route_prefix=f"/{app_name}")
        print(f"✅ GraphCast endpoint deployed successfully!")
        print(f"   Application: {app_name}")
        
        # Get the actual head node IP
        try:
            head_ip = get_node_ip_address()
        except:
            head_ip = "localhost"
        
        print(f"   API: http://<head-node-ip>:8000/{app_name}")
        print(f"   Model: {model_name}")
        print(f"   Autoscaling: Scale to zero enabled (5 min idle timeout)")
        print(f"   Schema endpoint: http://<head-node-ip>:8000/{app_name}/v1/schema")
        print(f"\n📊 Deployment Summary:")
        print(f"   Model: {model_name}")
        print(f"   Model path: {model_path}")
        print(f"   Application: {app_name}")
        print(f"   Endpoint: http://{head_ip}:8000/{app_name}")
        print(f"   Schema: http://{head_ip}:8000/{app_name}/v1/schema")
        print(f"\n📝 Example usage:")
        print(f"   # Get schema (discover inputs):")
        print(f"   curl http://{head_ip}:8000/{app_name}/v1/schema")
        print(f"   # Generate forecast:")
        print(f"   curl -X POST http://{head_ip}:8000/{app_name}/v1/forecast \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"latitude\": 40.7128, \"longitude\": -74.0060, \"forecast_hours\": 6}}'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy GraphCast weather forecasting model")
    parser.add_argument(
        "--model",
        type=str,
        default="shermansiu/dm_graphcast",
        help="Model name, local path, or HuggingFace model ID (default: shermansiu/dm_graphcast). Note: 'google-deepmind/graphcast' is not a HuggingFace model - it requires JAX setup from GitHub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model (overrides --model if provided)"
    )
    parser.add_argument(
        "--app-name",
        type=str,
        default=None,
        help="Application name (defaults to model name)"
    )
    
    args = parser.parse_args()
    
    model_path = args.model_path if args.model_path else args.model
    model_name = args.model.replace("/", "-").replace("_", "-")
    
    deploy_graphcast(model_name, model_path, args.app_name)


if __name__ == "__main__":
    main()
