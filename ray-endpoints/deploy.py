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
    
    # Start Ray Serve with HTTP options to listen on all interfaces
    # Note: HTTP options can only be set when Serve is first started
    # If Serve is already running, you need to restart it to change HTTP options
    try:
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(host="0.0.0.0", port=8000)
        serve.start(detached=True, http_options=http_options)
        print("✅ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
    except Exception as e:
        # Serve might already be running - check if we can get status
        try:
            status = serve.status()
            print(f"✅ Ray Serve is already running")
            print(f"   Note: To change HTTP host/port, restart Ray Serve first")
            print(f"   Current status: {len(status.applications) if hasattr(status, 'applications') else 'N/A'} applications")
        except:
            print(f"⚠️  Ray Serve may already be running: {e}")
            print("   If endpoints aren't accessible, you may need to restart Ray Serve")
            print("   with HTTP options: serve.start(http_options=HTTPOptions(host='0.0.0.0', port=8000))")
    
    # Deploy endpoints
    if args.endpoint:
        deploy_specific(args.endpoint)
    else:
        deploy_all()
    
    print("\n📋 Deployed endpoints:")
    try:
        # Get status to show deployed applications
        status = serve.status()
        if hasattr(status, 'applications') and status.applications:
            print("  Active applications:")
            for app_name, app_info in status.applications.items():
                route = getattr(app_info, 'route_prefix', 'N/A')
                print(f"    - {app_name}: {route}")
        else:
            # Try alternative method
            try:
                import requests
                head_node_ip = ray.util.get_node_ip_address()
                response = requests.get(f"http://{head_node_ip}:8265/api/serve/applications/", timeout=2)
                if response.status_code == 200:
                    apps = response.json()
                    if apps.get('applications'):
                        print("  Active applications (from API):")
                        for app_name, app_data in apps['applications'].items():
                            route = app_data.get('route_prefix', 'N/A')
                            print(f"    - {app_name}: {route}")
                    else:
                        print("  (No applications found via API)")
                else:
                    print("  (Could not fetch from API)")
            except:
                print("  (Use 'ray serve status' or check dashboard at http://<head-node-ip>:8265)")
    except Exception as e:
        print(f"  (Could not list deployments: {e})")
        print("  Use 'ray serve status' to see all deployments")
    
    # Get head node IP for user
    try:
        head_node_ip = ray.util.get_node_ip_address()
        print(f"\n🌐 Access endpoints at:")
        print(f"   http://{head_node_ip}:8000/onnx")
        print(f"   http://{head_node_ip}:8000/resnet50")
        print(f"   http://{head_node_ip}:8000/mobilenetv2")
        print(f"   http://{head_node_ip}:8000/stable-diffusion")
    except:
        print(f"\n🌐 Access endpoints at http://<head-node-ip>:8000")


if __name__ == "__main__":
    main()
