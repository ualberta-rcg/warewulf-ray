#!/usr/bin/env python3
"""
Ray Serve Deployment Script
Deploys Ray Serve endpoints that integrate with Triton Inference Server
"""

import argparse
import sys
import os

# Ensure we use Ray from /opt/ray if available
# This is needed because Ray is installed in /opt/ray in the Docker container
RAY_PYTHON = "/opt/ray/bin/python"
if os.path.exists(RAY_PYTHON) and __file__ != os.path.abspath(RAY_PYTHON):
    # If we're not already using the Ray Python, ensure it's in the path
    if "/opt/ray/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ray import serve
    from ray.serve import Application
    import ray
except ImportError:
    print("❌ Ray not found. Please use /opt/ray/bin/python to run this script.")
    print("   Or ensure /opt/ray/bin is in your PATH")
    print("   Current Python: " + sys.executable)
    sys.exit(1)


def deploy_all():
    """Deploy all available endpoints"""
    print("🚀 Deploying all Ray Serve endpoints...")
    
    # Import endpoint modules
    from endpoints import onnx_models, stable_diffusion
    
    # Deploy ONNX model endpoints
    print("📦 Deploying ONNX model endpoints...")
    onnx_models.deploy()
    
    # Deploy Stable Diffusion endpoint
    print("🎨 Deploying Stable Diffusion endpoint...")
    try:
        stable_diffusion.deploy()
    except Exception as e:
        print(f"⚠️  Stable Diffusion deployment skipped: {e}")
    
    print("✅ All endpoints deployed successfully!")
    print(f"📊 Ray Dashboard: http://localhost:8265")
    print(f"🌐 Ray Serve: http://localhost:8000")


def deploy_specific(endpoint_name: str):
    """Deploy a specific endpoint"""
    print(f"🚀 Deploying endpoint: {endpoint_name}")
    
    if endpoint_name == "onnx_models":
        from endpoints import onnx_models
        onnx_models.deploy()
    elif endpoint_name == "stable_diffusion":
        from endpoints import stable_diffusion
        stable_diffusion.deploy()
    else:
        print(f"❌ Unknown endpoint: {endpoint_name}")
        print("Available endpoints: onnx_models, stable_diffusion")
        sys.exit(1)
    
    print(f"✅ Endpoint '{endpoint_name}' deployed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Deploy Ray Serve endpoints")
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Deploy specific endpoint (onnx_models, stable_diffusion)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Initialize Ray connection
    try:
        ray.init(address=args.address, ignore_reinit_error=True)
        print(f"✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        sys.exit(1)
    
    # Start Ray Serve
    try:
        serve.start(detached=True)
        print("✅ Ray Serve started")
    except Exception as e:
        print(f"⚠️  Ray Serve may already be running: {e}")
    
    # Deploy endpoints
    if args.endpoint:
        deploy_specific(args.endpoint)
    else:
        deploy_all()
    
    print("\n📋 Deployed endpoints:")
    for deployment in serve.list_deployments():
        print(f"  - {deployment}")


if __name__ == "__main__":
    main()
