#!/usr/bin/env python3
"""
Ray Serve Deployment Script
Deploys Ray Serve endpoints that integrate with Triton Inference Server
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ray import serve
from ray.serve import Application
import ray


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
