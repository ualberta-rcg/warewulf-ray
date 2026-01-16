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
            # Try to connect to local Ray (should be running via systemd)
            ray.init(ignore_reinit_error=True)
            print("✓ Connected to local Ray")
        except Exception as e:
            print(f"⚠ Could not connect to Ray: {e}")
            print("  Ray should be running via systemd service")
            print("  Or set RAY_ADDRESS to connect to a remote cluster")
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
    
    # Start Ray Serve (will use existing Ray connection)
    try:
        serve.start(detached=True)
    except Exception as e:
        print(f"⚠ Ray Serve may already be running: {e}")
    
    deployments = []
    for endpoint_class in endpoint_classes:
        try:
            # Get autoscaling config (need to create instance temporarily, but don't keep it)
            temp_instance = endpoint_class()
            autoscaling_config = temp_instance.get_autoscaling_config()
            
            # Wrap FastAPI app in a Ray Serve deployment class
            # Create endpoint and app INSIDE __init__ to avoid serialization issues
            @serve.deployment(
                name=endpoint_class.DEPLOYMENT_NAME,
                autoscaling_config=autoscaling_config,
            )
            class FastAPIDeployment:
                def __init__(self):
                    # Create endpoint instance and app here (on worker side)
                    # This avoids serialization issues with FastAPI's thread locks
                    self.endpoint = endpoint_class()
                    self._app = self.endpoint.create_app()
                
                # Ray Serve will call this as an ASGI app
                async def __call__(self, scope, receive, send):
                    # Forward directly to FastAPI app (which is ASGI-compatible)
                    await self._app(scope, receive, send)
            
            # Deploy using serve.run with route_prefix
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
    
    try:
        serve.start(detached=True)
    except Exception as e:
        print(f"⚠ Ray Serve may already be running: {e}")
    
    # Get autoscaling config (need to create instance temporarily)
    temp_instance = endpoint_class()
    autoscaling_config = temp_instance.get_autoscaling_config()
    
    @serve.deployment(
        name=endpoint_class.DEPLOYMENT_NAME,
        autoscaling_config=autoscaling_config,
    )
    class FastAPIDeployment:
        def __init__(self):
            # Create endpoint instance and app here (on worker side)
            # This avoids serialization issues with FastAPI's thread locks
            self.endpoint = endpoint_class()
            self._app = self.endpoint.create_app()
        
        # Make the class itself callable as an ASGI app
        async def __call__(self, scope, receive, send):
            # Forward to FastAPI app
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
