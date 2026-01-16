"""
Deploy all discovered endpoints
"""
from ray import serve
import ray
from ray.serve.deployment import Deployment
from typing import List
import sys
import os
from pathlib import Path

# Handle both direct execution and module import
# Add current directory to path so we can import from it
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Import using relative imports (works when run from parent directory)
# or absolute imports (works when run directly)
try:
    from .loader import discover_endpoints, load_endpoint_from_file
    from .base import BaseEndpoint
except ImportError:
    # If relative import fails, import directly from current directory
    import loader
    import base
    discover_endpoints = loader.discover_endpoints
    load_endpoint_from_file = loader.load_endpoint_from_file
    BaseEndpoint = base.BaseEndpoint

def connect_to_ray():
    """Connect to Ray cluster if RAY_ADDRESS is set, otherwise use existing connection"""
    ray_address = os.environ.get("RAY_ADDRESS")
    
    if ray_address:
        print(f"📡 Connecting to Ray cluster at {ray_address}...")
        try:
            # Check if already connected
            if ray.is_initialized():
                print("✓ Already connected to Ray")
                return
            
            # Connect to remote cluster
            ray.init(address=ray_address, ignore_reinit_error=True)
            print(f"✓ Connected to Ray cluster at {ray_address}")
        except Exception as e:
            print(f"✗ Failed to connect to Ray cluster: {e}")
            print("  Make sure the Ray cluster is running and accessible")
            raise
    else:
        # Check if Ray is already initialized (from systemd service)
        try:
            if ray.is_initialized():
                print("✓ Using existing Ray connection (from systemd service)")
                return
            
            # Get the Ray address from the running cluster
            # Try to find it from ray status or session directory
            ray_address = None
            
            # Method 1: Try to get from ray status output
            try:
                import subprocess
                result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Look for the address in the output
                    # Ray status typically shows something like "ray://172.26.92.232:10001"
                    for line in result.stdout.split('\n'):
                        if 'ray://' in line:
                            # Extract address from ray:// format
                            import re
                            match = re.search(r'ray://([^:]+):(\d+)', line)
                            if match:
                                host = match.group(1)
                                port = match.group(2)
                                ray_address = f"{host}:{port}"
                                break
            except Exception:
                pass
            
            # Method 2: Get head node IP from system (for head node)
            if not ray_address:
                try:
                    import socket
                    import subprocess
                    # Get the primary IP address
                    # Try hostname -I first (more reliable)
                    result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0 and result.stdout.strip():
                        host_ip = result.stdout.strip().split()[0]
                        ray_address = f"{host_ip}:6379"
                    else:
                        # Fallback to gethostbyname
                        hostname = socket.gethostname()
                        host_ip = socket.gethostbyname(hostname)
                        ray_address = f"{host_ip}:6379"
                except Exception:
                    # Method 3: Use localhost with default port as last resort
                    ray_address = "localhost:6379"
            
            # Now connect to the existing cluster
            print(f"📡 Connecting to existing Ray cluster at {ray_address}...")
            ray.init(address=ray_address, ignore_reinit_error=True)
            print(f"✓ Connected to Ray cluster at {ray_address}")
        except Exception as e:
            print(f"⚠ Could not connect to Ray: {e}")
            print("  Ray should be running via systemd service")
            print("  Or set RAY_ADDRESS to connect to a remote cluster")
            print("  Example: export RAY_ADDRESS=172.26.92.232:6379")
            raise

def deploy_all_endpoints():
    """Discover and deploy all endpoints"""
    # Connect to Ray first (should already be running via systemd)
    connect_to_ray()
    
    print("Discovering endpoints...")
    endpoint_classes = discover_endpoints()
    
    if not endpoint_classes:
        print("No endpoints found!")
        return
    
    print(f"Found {len(endpoint_classes)} endpoints")
    
    # Start Ray Serve with HTTP options to listen on all interfaces (0.0.0.0)
    # This allows access from outside the head node
    # HTTP options are cluster-scoped and can't be changed at runtime, so we need to restart
    try:
        # Check if Serve is already running and shutdown if needed
        try:
            serve.start(detached=True)
            # If we get here, Serve was already running - shutdown to apply new config
            print("⚠ Ray Serve already running, restarting with new HTTP config...")
            serve.shutdown()
            import time
            time.sleep(2)  # Wait for shutdown to complete
        except Exception:
            # Serve wasn't running, that's fine
            pass
        
        # Now start with correct HTTP options
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(
            host="0.0.0.0",  # Listen on all interfaces
            port=8000,
        )
        serve.start(detached=True, http_options=http_options)
        print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
    except Exception as e:
        # If HTTPOptions doesn't exist or serve is already running, try alternative methods
        try:
            # Try with dict format
            serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
            print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
        except Exception as e2:
            # If that fails, just start normally (may already be running)
            try:
                serve.start(detached=True)
                print("⚠ Ray Serve started (may need to configure host manually)")
            except Exception as e3:
                print(f"⚠ Ray Serve may already be running: {e3}")
    
    # Clean up any stale/failed deployments
    try:
        status = serve.status()
        if hasattr(status, 'applications') and status.applications:
            for app_name, app_info in status.applications.items():
                # Check for failed apps or apps with failed deployments
                if hasattr(app_info, 'status'):
                    if app_info.status in ["DEPLOYING_FAILED", "UNHEALTHY"]:
                        print(f"⚠ Found failed app: {app_name}, attempting to delete...")
                        try:
                            serve.delete(app_name)
                            print(f"✓ Deleted failed app: {app_name}")
                        except Exception as e:
                            print(f"  Could not delete {app_name}: {e}")
                # Also check individual deployments
                if hasattr(app_info, 'deployments'):
                    for dep_name, dep_info in app_info.deployments.items():
                        if hasattr(dep_info, 'status') and dep_info.status in ["UNHEALTHY", "DEPLOYING_FAILED"]:
                            print(f"⚠ Found failed deployment: {dep_name} in app {app_name}")
    except Exception as e:
        print(f"⚠ Could not check for stale deployments: {e}")
    
    deployments = []
    for endpoint_class in endpoint_classes:
        try:
            # Get autoscaling config (need to create instance temporarily, but don't keep it)
            temp_instance = endpoint_class()
            autoscaling_config = temp_instance.get_autoscaling_config()
            
            # Wrap FastAPI app in a Ray Serve deployment class
            # Create endpoint and app INSIDE __init__ to avoid serialization issues
            # Add runtime_env to ensure dependencies and paths are available on workers
            import os
            from pathlib import Path
            
            # Get the ray-endpoints directory path
            ray_endpoints_dir = Path(__file__).parent.absolute()
            
            # Support Compute Canada modules if available
            # Check if module command is available and load common modules
            compute_canada_modules = []
            if os.path.exists("/cvmfs/soft.computecanada.ca/config/profile/bash.sh"):
                # Compute Canada modules available
                # You can add module loading here if needed
                # For example: compute_canada_modules = ["python/3.12", "cuda/12.0"]
                pass
            
            # Setup runtime environment for workers
            # This ensures the base module and dependencies are available on worker nodes
            
            # Check for shared venv on NFS (more efficient than installing on each worker)
            shared_venv_path = "/data/ray-endpoints-venv"
            use_shared_venv = os.path.exists(shared_venv_path) and os.path.exists(f"{shared_venv_path}/bin/python")
            
            # Build PATH with shared venv if available
            path_components = [os.environ.get("PATH", "")]
            if use_shared_venv:
                path_components.insert(0, f"{shared_venv_path}/bin")
            
            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": str(ray_endpoints_dir),
                    "PATH": ":".join(path_components),
                },
                # Add pip packages if needed (uncomment and add packages)
                # Note: If using shared venv, install packages there instead:
                # /data/ray-endpoints-venv/bin/pip install package_name
                # "pip": ["psutil", "requests"],  # Only use if NOT using shared venv
            }
            
            # Store modules list and venv info for use in __init__
            modules_to_load = compute_canada_modules
            shared_venv_available = use_shared_venv
            
            @serve.deployment(
                name=endpoint_class.DEPLOYMENT_NAME,
                autoscaling_config=autoscaling_config,
                runtime_env=runtime_env,
            )
            class FastAPIDeployment:
                def __init__(self):
                    # Ensure Python path is set for imports (critical for worker nodes)
                    import sys
                    ray_endpoints_dir = Path(__file__).parent.parent.absolute()
                    if str(ray_endpoints_dir) not in sys.path:
                        sys.path.insert(0, str(ray_endpoints_dir))
                    
                    # Activate shared venv if available (NFS share)
                    if shared_venv_available:
                        try:
                            shared_venv_path = "/data/ray-endpoints-venv"
                            # Add venv to sys.path
                            venv_site_packages = f"{shared_venv_path}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
                            if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
                                sys.path.insert(0, venv_site_packages)
                            print(f"✓ Using shared venv from NFS: {shared_venv_path}")
                        except Exception as e:
                            print(f"⚠ Could not use shared venv: {e}")
                    
                    # Load Compute Canada modules if available
                    if modules_to_load:
                        try:
                            import subprocess
                            # Source the module system and load modules
                            module_cmd = "source /cvmfs/soft.computecanada.ca/config/profile/bash.sh && " + \
                                       " && ".join([f"module load {m}" for m in modules_to_load])
                            subprocess.run(["bash", "-c", module_cmd], check=True)
                            print(f"✓ Loaded Compute Canada modules: {modules_to_load}")
                        except Exception as e:
                            print(f"⚠ Could not load Compute Canada modules: {e}")
                    
                    # Create endpoint instance and app here (on worker side)
                    # This avoids serialization issues with FastAPI's thread locks
                    self.endpoint = endpoint_class()
                    self._app = self.endpoint.create_app()
                
                # Ray Serve will call this as an ASGI app
                # When route_prefix is set, Ray Serve handles routing automatically
                # We forward the ASGI interface directly to FastAPI
                async def __call__(self, scope, receive, send):
                    # Forward directly to FastAPI app (which is ASGI-compatible)
                    # Ray Serve strips the route_prefix before forwarding, so paths are correct
                    await self._app(scope, receive, send)
            
            # Deploy using serve.run with route_prefix
            # In Ray Serve 2.53.0, route_prefix properly routes to the deployment
            serve.run(
                FastAPIDeployment.bind(),
                name=endpoint_class.DEPLOYMENT_NAME,
                route_prefix=endpoint_class.ROUTE_PREFIX,
            )
            deployments.append(endpoint_class.DEPLOYMENT_NAME)
            print(f"✓ Deployed {endpoint_class.DEPLOYMENT_NAME} at {endpoint_class.ROUTE_PREFIX}")
        except Exception as e:
            print(f"✗ Failed to deploy {endpoint_class.DEPLOYMENT_NAME}: {e}")
            import traceback
            traceback.print_exc()
    
    return deployments

def deploy_single_endpoint(endpoint_file: str):
    """Deploy a single endpoint from a file"""
    # Connect to Ray first (should already be running via systemd)
    connect_to_ray()
    
    endpoint_class = load_endpoint_from_file(endpoint_file)
    
    # Start Ray Serve with HTTP options to listen on all interfaces (0.0.0.0)
    # HTTP options are cluster-scoped and can't be changed at runtime, so we need to restart
    try:
        # Check if Serve is already running and shutdown if needed
        try:
            serve.start(detached=True)
            # If we get here, Serve was already running - shutdown to apply new config
            print("⚠ Ray Serve already running, restarting with new HTTP config...")
            serve.shutdown()
            import time
            time.sleep(2)  # Wait for shutdown to complete
        except Exception:
            # Serve wasn't running, that's fine
            pass
        
        # Now start with correct HTTP options
        from ray.serve.config import HTTPOptions
        http_options = HTTPOptions(
            host="0.0.0.0",  # Listen on all interfaces
            port=8000,
        )
        serve.start(detached=True, http_options=http_options)
        print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
    except Exception as e:
        # If HTTPOptions doesn't exist or serve is already running, try alternative methods
        try:
            # Try with dict format
            serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
            print("✓ Ray Serve started on 0.0.0.0:8000 (accessible from network)")
        except Exception as e2:
            # If that fails, just start normally (may already be running)
            try:
                serve.start(detached=True)
                print("⚠ Ray Serve started (may need to configure host manually)")
            except Exception as e3:
                print(f"⚠ Ray Serve may already be running: {e3}")
    
    # Get autoscaling config (need to create instance temporarily)
    temp_instance = endpoint_class()
    autoscaling_config = temp_instance.get_autoscaling_config()
    
    # Setup runtime environment for workers
    import os
    from pathlib import Path
    
    # Get the ray-endpoints directory path
    ray_endpoints_dir = Path(__file__).parent.absolute()
    
    # Setup runtime environment for workers
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": str(ray_endpoints_dir),
            "PATH": os.environ.get("PATH", ""),
        },
        # Add pip packages if needed (uncomment and add packages)
        # "pip": ["psutil", "requests"],  # Add any missing packages here
    }
    
    # Support Compute Canada modules if available
    compute_canada_modules = []
    if os.path.exists("/cvmfs/soft.computecanada.ca/config/profile/bash.sh"):
        # Compute Canada modules available - add modules here if needed
        pass
    
    @serve.deployment(
        name=endpoint_class.DEPLOYMENT_NAME,
        autoscaling_config=autoscaling_config,
        runtime_env=runtime_env,
    )
    class FastAPIDeployment:
        def __init__(self):
            # Ensure Python path is set for imports (critical for worker nodes)
            import sys
            ray_endpoints_dir = Path(__file__).parent.parent.absolute()
            if str(ray_endpoints_dir) not in sys.path:
                sys.path.insert(0, str(ray_endpoints_dir))
            
            # Load Compute Canada modules if available
            if compute_canada_modules:
                try:
                    import subprocess
                    for module in compute_canada_modules:
                        subprocess.run(["module", "load", module], check=True, shell=True)
                except Exception as e:
                    print(f"⚠ Could not load Compute Canada modules: {e}")
            
            # Create endpoint instance and app here (on worker side)
            # This avoids serialization issues with FastAPI's thread locks
            self.endpoint = endpoint_class()
            self._app = self.endpoint.create_app()
        
        # Ray Serve will call this as an ASGI app
        async def __call__(self, scope, receive, send):
            # Forward directly to FastAPI app (which is ASGI-compatible)
            await self._app(scope, receive, send)
    
    serve.run(
        FastAPIDeployment.bind(),
        name=endpoint_class.DEPLOYMENT_NAME,
        route_prefix=endpoint_class.ROUTE_PREFIX,
    )
    print(f"✓ Deployed {endpoint_class.DEPLOYMENT_NAME}")
    return endpoint_class.DEPLOYMENT_NAME

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Deploy specific file
        deploy_single_endpoint(sys.argv[1])
    else:
        # Deploy all
        deploy_all_endpoints()
