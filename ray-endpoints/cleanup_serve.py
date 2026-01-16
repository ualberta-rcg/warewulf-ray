#!/usr/bin/env python3
"""
Cleanup script for Ray Serve - removes stale/failed deployments
"""
from ray import serve
import ray
import sys
import os

def cleanup_serve():
    """Clean up stale Ray Serve deployments"""
    # Connect to Ray
    ray_address = os.environ.get("RAY_ADDRESS")
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        # Get the Ray address from the running cluster
        if ray.is_initialized():
            pass  # Already connected
        else:
            # Try to get address from system
            try:
                import subprocess
                import socket
                # Get head node IP
                result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    host_ip = result.stdout.strip().split()[0]
                    ray_address = f"{host_ip}:6379"
                else:
                    hostname = socket.gethostname()
                    host_ip = socket.gethostbyname(hostname)
                    ray_address = f"{host_ip}:6379"
                ray.init(address=ray_address, ignore_reinit_error=True)
            except Exception:
                # Fallback to localhost
                ray.init(address="localhost:6379", ignore_reinit_error=True)
    
    try:
        serve.start(detached=True)
    except Exception as e:
        print(f"⚠ Ray Serve may already be running: {e}")
    
    print("Checking Ray Serve status...")
    try:
        status = serve.status()
        print(f"Ray Serve is running")
        
        # Check applications
        if hasattr(status, 'applications') and status.applications:
            print(f"\nFound {len(status.applications)} application(s):")
            for app_name, app_info in status.applications.items():
                print(f"\n  App: {app_name}")
                if hasattr(app_info, 'status'):
                    print(f"    Status: {app_info.status}")
                if hasattr(app_info, 'message'):
                    print(f"    Message: {app_info.message}")
                
                # Check if we should delete this app
                should_delete = False
                if hasattr(app_info, 'status'):
                    if app_info.status in ["DEPLOYING_FAILED", "UNHEALTHY"]:
                        should_delete = True
                        print(f"    ⚠ Marked for deletion (status: {app_info.status})")
                
                # Check deployments within app
                if hasattr(app_info, 'deployments'):
                    print(f"    Deployments:")
                    for dep_name, dep_info in app_info.deployments.items():
                        dep_status = getattr(dep_info, 'status', 'unknown')
                        print(f"      - {dep_name}: {dep_status}")
                        if dep_status in ["UNHEALTHY", "DEPLOYING_FAILED"]:
                            should_delete = True
                            print(f"        ⚠ Marked for deletion")
                
                # Delete if needed
                if should_delete:
                    try:
                        print(f"    🗑️  Deleting app: {app_name}...")
                        serve.delete(app_name)
                        print(f"    ✓ Deleted {app_name}")
                    except Exception as e:
                        print(f"    ✗ Could not delete {app_name}: {e}")
        else:
            print("  No applications found")
            
    except Exception as e:
        print(f"✗ Error checking status: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✓ Cleanup complete!")
    return 0

if __name__ == "__main__":
    sys.exit(cleanup_serve())
