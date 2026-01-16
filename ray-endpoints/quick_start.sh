#!/bin/bash
# Quick start script to deploy all endpoints
# Note: Ray should already be running via systemd service

set -e

echo "🚀 Starting Ray Serve endpoints deployment..."

# Check if cleanup flag is set
CLEANUP=${1:-""}
if [ "$CLEANUP" = "--cleanup" ] || [ "$CLEANUP" = "-c" ]; then
    echo "🧹 Cleaning up before deployment..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/cleanup_ray_serve.sh" ]; then
        bash "$SCRIPT_DIR/cleanup_ray_serve.sh"
        echo ""
        sleep 2
    else
        echo "⚠ Cleanup script not found, skipping cleanup"
    fi
fi

# Check if Ray is available
if ! command -v ray &> /dev/null; then
    echo "❌ Ray not found. Make sure Ray is in PATH."
    exit 1
fi

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify NFS setup (quick check)
echo "🔍 Verifying NFS setup..."
if [ -f "$SCRIPT_DIR/verify_nfs_setup.sh" ]; then
    if ! bash "$SCRIPT_DIR/verify_nfs_setup.sh" 2>/dev/null; then
        echo "⚠ NFS setup verification had issues"
        echo "   Continuing anyway, but workers may not find base.py"
        echo "   Run 'bash verify_nfs_setup.sh' for details"
        echo ""
    fi
fi

# Check if RAY_ADDRESS is set (remote cluster)
if [ -n "$RAY_ADDRESS" ]; then
    echo "📡 RAY_ADDRESS is set to: $RAY_ADDRESS"
    echo "   Connecting to remote Ray cluster..."
    
    # Test connection
    if ! ray status --address="$RAY_ADDRESS" &> /dev/null; then
        echo "❌ Cannot connect to Ray cluster at $RAY_ADDRESS"
        echo "   Make sure the cluster is running and accessible"
        exit 1
    fi
    echo "✓ Connected to remote Ray cluster"
else
    # Check if local Ray is running (via systemd)
    if ! ray status &> /dev/null; then
        echo "❌ Ray is not running!"
        echo "   Start Ray with: systemctl start ray-worker (or ray-head)"
        echo "   Or set RAY_ADDRESS to connect to a remote cluster"
        exit 1
    else
        echo "✓ Ray cluster is running (via systemd)"
    fi
fi

# Deploy endpoints
echo "📦 Deploying endpoints..."
# Run deploy.py from the ray-endpoints directory
cd "$SCRIPT_DIR"

# Use the Ray Python from venv if available
if [ -f "/opt/ray/bin/python" ]; then
    PYTHON="/opt/ray/bin/python"
else
    PYTHON="python3"
fi

$PYTHON deploy.py

echo ""
echo "✅ Deployment complete!"
echo ""

# Get the head node IP address for network access
HEAD_IP=$(hostname -I | awk '{print $1}')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="localhost"
fi

echo "Endpoints available:"
echo "  - Stable Diffusion:"
echo "    Local:  http://localhost:8000/api/v1/stable-diffusion"
echo "    Network: http://${HEAD_IP}:8000/api/v1/stable-diffusion"
echo "  - Kandinsky3:"
echo "    Local:  http://localhost:8000/api/v1/kandinsky3"
echo "    Network: http://${HEAD_IP}:8000/api/v1/kandinsky3"
echo ""
echo "Test with:"
echo "  curl http://localhost:8000/api/v1/stable-diffusion/health"
echo "  curl http://${HEAD_IP}:8000/api/v1/stable-diffusion/health"
echo "  curl http://localhost:8000/api/v1/kandinsky3/health"
echo "  curl http://${HEAD_IP}:8000/api/v1/kandinsky3/health"
echo ""
echo "To cleanup and redeploy:"
echo "  bash quick_start.sh --cleanup"
echo ""
echo "To check routes:"
echo "  curl http://localhost:8000/-/routes"
echo ""
echo "Verifying deployment..."
$PYTHON << 'EOF' || echo "⚠ Verification had issues, but deployment may still be successful"
import ray
from ray import serve
import sys
import time

try:
    # Get Ray address from system
    import subprocess
    import socket
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            host_ip = result.stdout.strip().split()[0]
            ray_address = f"{host_ip}:6379"
        else:
            hostname = socket.gethostname()
            host_ip = socket.gethostbyname(hostname)
            ray_address = f"{host_ip}:6379"
    except Exception:
        ray_address = "localhost:6379"
    
    ray.init(address=ray_address, ignore_reinit_error=True)
    serve.start(detached=True)
    
    # Wait a moment for deployments to register
    time.sleep(2)
    
    status = serve.status()
    print("\n=== Deployment Status ===")
    
    if hasattr(status, 'applications') and status.applications:
        print(f"✓ Found {len(status.applications)} application(s):")
        for app_name, app_info in status.applications.items():
            print(f"\n  App: {app_name}")
            if hasattr(app_info, 'status'):
                print(f"    Status: {app_info.status}")
            if hasattr(app_info, 'deployments'):
                for dep_name, dep_info in app_info.deployments.items():
                    print(f"    Deployment: {dep_name}")
                    if hasattr(dep_info, 'status'):
                        print(f"      Status: {dep_info.status}")
                    if hasattr(dep_info, 'replicas'):
                        replicas = dep_info.replicas
                        print(f"      Replicas: {replicas}")
                        if replicas == 0:
                            print(f"      ⚠ Scaled to zero - will scale up on first request")
    else:
        print("⚠ No applications found - deployments may still be starting")
        print("  Wait a few seconds and check again")
    
    # Try to get routes
    try:
        import requests
        response = requests.get("http://localhost:8000/-/routes", timeout=3)
        if response.status_code == 200:
            routes_text = response.text.strip()
            if routes_text and routes_text != "Route table is not populated yet.":
                print(f"\n✓ Routes available:")
                try:
                    import json
                    routes = response.json()
                    for route, info in routes.items():
                        print(f"  {route}")
                except:
                    print(f"  {routes_text}")
            else:
                print(f"\n⚠ Route table not populated yet (deployments may be scaling up)")
        else:
            print(f"\n⚠ Could not fetch routes (HTTP {response.status_code})")
    except ImportError:
        print(f"\n⚠ requests library not available - install with: /opt/ray/bin/pip install requests")
    except Exception as e:
        print(f"\n⚠ Could not check routes: {e}")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Error verifying deployment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
