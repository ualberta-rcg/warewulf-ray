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
    
    # Deploy ONNX model endpoints (simple version following Ray docs)
    print("📦 Deploying ONNX model endpoints...")
    try:
        from endpoints import onnx_models_simple
        onnx_models_simple.deploy()
    except ImportError:
        # Fallback to original if simple version not available
        from endpoints import onnx_models
        onnx_models.deploy()
    
    # Import stable diffusion
    from endpoints import stable_diffusion
    
    # Deploy Stable Diffusion endpoint
    print("🎨 Deploying Stable Diffusion endpoint...")
    try:
        stable_diffusion.deploy()
    except Exception as e:
        print(f"⚠️  Stable Diffusion deployment skipped: {e}")
    
    print("✅ All endpoints deployed successfully!")
    try:
        head_node_ip = ray.util.get_node_ip_address()
        print(f"📊 Ray Dashboard: http://{head_node_ip}:8265")
        print(f"🌐 Ray Serve: http://{head_node_ip}:8001 (Triton is on port 8000)")
    except:
        print(f"📊 Ray Dashboard: http://<head-node-ip>:8265")
        print(f"🌐 Ray Serve: http://<head-node-ip>:8001 (Triton is on port 8000)")


def deploy_specific(endpoint_name: str):
    """Deploy a specific endpoint"""
    print(f"🚀 Deploying endpoint: {endpoint_name}")
    
    if endpoint_name == "onnx_models":
        try:
            from endpoints import onnx_models_simple
            onnx_models_simple.deploy()
        except ImportError:
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
    
    # Check Triton setup - we use standalone Triton binary (recommended for Warewulf)
    # The standalone binary is available via nvidia-pytriton at:
    # /opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver
    print("🔍 Checking Triton setup...")
    
    TRITON_BINARY = "/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver"
    TRITON_URL = "localhost:8000"
    
    # Check if Triton binary exists
    if os.path.exists(TRITON_BINARY):
        print(f"✅ Triton binary found: {TRITON_BINARY}")
        print("   Using standalone Triton binary (recommended for Warewulf)")
    else:
        print(f"⚠️  Triton binary not found at {TRITON_BINARY}")
        print("   Make sure nvidia-pytriton is installed:")
        print("   /opt/ray/bin/pip install nvidia-pytriton")
    
    # Check if Triton server is already running
    try:
        import requests
        response = requests.get(f"http://{TRITON_URL}/v2/health/live", timeout=2)
        if response.status_code == 200:
            print(f"✅ Triton server is already running on {TRITON_URL}")
        else:
            print(f"⚠️  Triton server not responding on {TRITON_URL}")
            print("   Will start Triton automatically if binary is available")
    except Exception as e:
        print(f"⚠️  Triton server not running on {TRITON_URL}")
        print("   Will use HTTP client mode (Triton must be started separately)")
        print("   To start Triton: bash start_triton.sh")
    
    # Try to start Triton if not running and binary is available
    if os.path.exists(TRITON_BINARY):
        try:
            import requests
            response = requests.get(f"http://{TRITON_URL}/v2/health/live", timeout=2)
            if response.status_code != 200:
                # Triton not running - try to start it
                print("🚀 Starting Triton server...")
                import subprocess
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                start_script = os.path.join(script_dir, "start_triton.sh")
                if os.path.exists(start_script):
                    # Start in background (non-blocking)
                    subprocess.Popen(
                        ["bash", start_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print("   (Triton starting in background - may take a few seconds)")
                else:
                    print(f"   (start_triton.sh not found at {start_script})")
        except Exception as e:
            print(f"   (Could not auto-start Triton: {e})")
            print("   Start manually with: bash start_triton.sh")
    
    # Start Ray Serve with HTTP options to listen on all interfaces
    # Note: Ray Serve uses port 8001 to avoid conflict with Triton (port 8000)
    # HTTP options can only be set when Serve is first started
    # If Serve is already running, you need to restart it to change HTTP options
    RAY_SERVE_PORT = 8001  # Use 8001 to avoid conflict with Triton on 8000
    try:
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
        serve.start(detached=True, http_options=http_options)
        print(f"✅ Ray Serve started on 0.0.0.0:{RAY_SERVE_PORT} (accessible from network)")
        print(f"   Note: Triton is on port 8000, Ray Serve is on port {RAY_SERVE_PORT}")
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
            print(f"   with HTTP options: serve.start(http_options=HTTPOptions(host='0.0.0.0', port={RAY_SERVE_PORT}))")
    
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
    RAY_SERVE_PORT = 8001  # Match the port used above
    try:
        head_node_ip = ray.util.get_node_ip_address()
        print(f"\n🌐 Access endpoints at:")
        print(f"   http://{head_node_ip}:{RAY_SERVE_PORT}/onnx")
        print(f"   http://{head_node_ip}:{RAY_SERVE_PORT}/resnet50")
        print(f"   http://{head_node_ip}:{RAY_SERVE_PORT}/mobilenetv2")
        print(f"   http://{head_node_ip}:{RAY_SERVE_PORT}/stable-diffusion")
        print(f"\n   Triton: http://{head_node_ip}:8000 (model inference)")
        print(f"   Ray Serve: http://{head_node_ip}:{RAY_SERVE_PORT} (HTTP API)")
        print(f"\n🧪 Test endpoints with:")
        print(f"   chmod +x test_endpoints.sh && HEAD_NODE_IP={head_node_ip} RAY_SERVE_PORT={RAY_SERVE_PORT} ./test_endpoints.sh")
        print(f"   Or use curl:")
        print(f"   curl -X POST \"http://{head_node_ip}:{RAY_SERVE_PORT}/resnet50\" \\")
        print(f"        -H \"Content-Type: image/jpeg\" \\")
        print(f"        --data-binary @boot.jpg")
    except:
        print(f"\n🌐 Access endpoints at http://<head-node-ip>:8001")
        print(f"   (Triton is on port 8000, Ray Serve is on port 8001)")


if __name__ == "__main__":
    main()
