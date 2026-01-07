"""
Deploy all discovered endpoints
"""
from ray import serve
from ray.serve.deployment import Deployment
from typing import List
import sys
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

def deploy_all_endpoints():
    """Discover and deploy all endpoints"""
    print("Discovering endpoints...")
    endpoint_classes = discover_endpoints()
    
    if not endpoint_classes:
        print("No endpoints found!")
        return
    
    print(f"Found {len(endpoint_classes)} endpoints")
    
    # Start Ray Serve
    serve.start(detached=True)
    
    deployments = []
    for endpoint_class in endpoint_classes:
        try:
            # Create endpoint instance
            endpoint_instance = endpoint_class()
            app = endpoint_instance.create_app()
            
            # Wrap FastAPI app in a Ray Serve deployment class
            # Ray Serve expects a class with __call__ that returns responses
            @serve.deployment(
                name=endpoint_class.DEPLOYMENT_NAME,
                autoscaling_config=endpoint_instance.get_autoscaling_config(),
            )
            class FastAPIDeployment:
                def __init__(self):
                    self.endpoint = endpoint_instance
                    self._app = app
                
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
    endpoint_class = load_endpoint_from_file(endpoint_file)
    
    serve.start(detached=True)
    
    endpoint_instance = endpoint_class()
    app = endpoint_instance.create_app()
    
    @serve.deployment(
        name=endpoint_class.DEPLOYMENT_NAME,
        autoscaling_config=endpoint_instance.get_autoscaling_config(),
    )
    class FastAPIDeployment:
        def __init__(self):
            self.endpoint = endpoint_instance
            self._app = app
        
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
