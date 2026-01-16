"""
Deploy Triton-based endpoints
These are standalone deployments that work better than BaseEndpoint ones
"""
from ray import serve
import ray
from ray.serve.config import HTTPOptions
import os
import sys
from pathlib import Path

# Add endpoints directory to path
_endpoints_dir = Path(__file__).parent / "endpoints"
if str(_endpoints_dir) not in sys.path:
    sys.path.insert(0, str(_endpoints_dir))

def connect_to_ray():
    """Connect to Ray cluster"""
    ray_address = os.environ.get("RAY_ADDRESS")
    
    if ray_address:
        print(f"📡 Connecting to Ray cluster at {ray_address}...")
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"✓ Connected to Ray cluster at {ray_address}")
    else:
        # Get head node IP from system
        try:
            import subprocess
            import socket
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
        
        print(f"📡 Connecting to existing Ray cluster at {ray_address}...")
        ray.init(address=ray_address, ignore_reinit_error=True)
        print(f"✓ Connected to Ray cluster at {ray_address}")

def start_serve():
    """Start Ray Serve with HTTP options"""
    try:
        # Check if Serve is already running
        try:
            serve.start(detached=True)
            print("⚠ Ray Serve already running, restarting with new HTTP config...")
            serve.shutdown()
            import time
            time.sleep(2)
        except Exception:
            pass
        
        # Start with HTTP options
        try:
            http_options = HTTPOptions(host="0.0.0.0", port=8000)
            serve.start(detached=True, http_options=http_options)
            print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
        except Exception:
            try:
                serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
                print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
            except Exception:
                serve.start(detached=True)
                print("⚠ Ray Serve started (may need to configure host manually)")
    except Exception as e:
        print(f"⚠ Ray Serve may already be running: {e}")

def deploy_triton_endpoints():
    """Deploy all Triton endpoints"""
    # Connect to Ray
    connect_to_ray()
    
    # Start Serve
    start_serve()
    
    # Import and deploy endpoints
    print("")
    print("📦 Deploying Triton endpoints...")
    
    # Deploy Stable Diffusion Triton
    print("")
    print("🚀 Deploying Stable Diffusion Triton...")
    try:
        from stable_diffusion_triton import StableDiffusionTriton
        serve.run(
            StableDiffusionTriton.bind(),
            name="stable-diffusion-triton",
            route_prefix="/api/v1/stable-diffusion-triton"
        )
        print("✓ Stable Diffusion Triton deployed at /api/v1/stable-diffusion-triton")
    except Exception as e:
        print(f"✗ Failed to deploy Stable Diffusion Triton: {e}")
        import traceback
        traceback.print_exc()
    
    # Deploy Kandinsky3 Triton
    print("")
    print("🚀 Deploying Kandinsky3 Triton...")
    try:
        from kandinsky3_triton import Kandinsky3Triton
        serve.run(
            Kandinsky3Triton.bind(),
            name="kandinsky3-triton",
            route_prefix="/api/v1/kandinsky3-triton"
        )
        print("✓ Kandinsky3 Triton deployed at /api/v1/kandinsky3-triton")
    except Exception as e:
        print(f"✗ Failed to deploy Kandinsky3 Triton: {e}")
        import traceback
        traceback.print_exc()
    
    print("")
    print("✅ Deployment complete!")
    
    # Get head node IP
    try:
        import subprocess
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            head_ip = result.stdout.strip().split()[0]
        else:
            head_ip = "localhost"
    except Exception:
        head_ip = "localhost"
    
    print("")
    print("Endpoints available:")
    print("  - Stable Diffusion Triton:")
    print("    Local:  http://localhost:8000/api/v1/stable-diffusion-triton")
    print("    Network: http://{}:8000/api/v1/stable-diffusion-triton".format(head_ip))
    print("  - Kandinsky3 Triton:")
    print("    Local:  http://localhost:8000/api/v1/kandinsky3-triton")
    print("    Network: http://{}:8000/api/v1/kandinsky3-triton".format(head_ip))
    print("")
    print("Test with:")
    print("  curl http://localhost:8000/api/v1/stable-diffusion-triton/health")
    print("  curl http://{}:8000/api/v1/stable-diffusion-triton/health".format(head_ip))
    print("  curl http://localhost:8000/api/v1/kandinsky3-triton/health")
    print("  curl http://{}:8000/api/v1/kandinsky3-triton/health".format(head_ip))

if __name__ == "__main__":
    deploy_triton_endpoints()
