#!/usr/bin/env python3
"""
Clean up failed Ray Serve deployments
"""

import sys
import os

# Ensure we use Ray from /opt/ray
RAY_PYTHON = "/opt/ray/bin/python"
if os.path.exists(RAY_PYTHON) and __file__ != os.path.abspath(RAY_PYTHON):
    if "/opt/ray/bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ray import serve
    import ray
except ImportError:
    print("❌ Ray not found. Please use /opt/ray/bin/python to run this script.")
    sys.exit(1)


def main():
    """Clean up failed deployments"""
    
    print("🧹 Cleaning up failed Ray Serve deployments...")
    
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print("✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        sys.exit(1)
    
    try:
        # Get current status
        status = serve.status()
        print(f"\n📊 Current deployments:")
        
        # List all applications
        if hasattr(status, 'applications'):
            for app_name, app_info in status.applications.items():
                print(f"   Application: {app_name}")
                if hasattr(app_info, 'deployments'):
                    for dep_name, dep_info in app_info.deployments.items():
                        print(f"     - {dep_name}: {dep_info.status if hasattr(dep_info, 'status') else 'unknown'}")
        
        # Delete the default application if it has failed deployments
        try:
            print("\n🗑️  Deleting failed 'default' application...")
            serve.delete("default")
            print("✅ Deleted 'default' application")
        except Exception as e:
            print(f"⚠️  Could not delete 'default' application: {e}")
            print("   (This is OK if it doesn't exist)")
        
        print("\n✅ Cleanup complete!")
        print("   You can now redeploy with: bash deploy_hf.sh")
        
    except Exception as e:
        print(f"⚠️  Error during cleanup: {e}")
        print("   Trying to continue anyway...")
    
    print("\n💡 To verify, check status with:")
    print("   /opt/ray/bin/python -c \"from ray import serve; import ray; ray.init(address='auto'); print(serve.status())\"")


if __name__ == "__main__":
    main()
