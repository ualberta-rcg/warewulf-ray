"""
Deploy all discovered endpoints
"""
from ray import serve
from ray.serve.deployment import Deployment
from .loader import discover_endpoints, load_endpoint_from_file
from .base import BaseEndpoint
from typing import List
import sys

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
            # Create the endpoint instance to get the app
            endpoint_instance = endpoint_class()
            app = endpoint_instance.create_app()
            
            # Create deployment
            deployment = serve.deployment(
                endpoint_instance,
                name=endpoint_class.DEPLOYMENT_NAME,
                route_prefix=endpoint_class.ROUTE_PREFIX,
                autoscaling_config=endpoint_instance.get_autoscaling_config(),
            )
            
            # Deploy
            deployment.deploy()
            deployments.append(deployment)
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
    
    deployment = serve.deployment(
        endpoint_instance,
        name=endpoint_class.DEPLOYMENT_NAME,
        route_prefix=endpoint_class.ROUTE_PREFIX,
        autoscaling_config=endpoint_instance.get_autoscaling_config(),
    )
    
    deployment.deploy()
    print(f"✓ Deployed {endpoint_class.DEPLOYMENT_NAME}")
    return deployment

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Deploy specific file
        deploy_single_endpoint(sys.argv[1])
    else:
        # Deploy all
        deploy_all_endpoints()
