#!/usr/bin/env python3
"""
Delete a specific Ray Serve application by name
"""

import sys
import os
import argparse

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
    parser = argparse.ArgumentParser(description="Delete a Ray Serve application")
    parser.add_argument(
        "app_name",
        type=str,
        nargs="?",
        help="Application name to delete (e.g., 'medgemma-medgemma-27b-it')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all applications instead of deleting",
    )
    
    args = parser.parse_args()
    
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print("✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        sys.exit(1)
    
    try:
        status = serve.status()
        
        if args.list or not args.app_name:
            # List all applications
            print("\n📊 Current Ray Serve applications:")
            if hasattr(status, 'applications') and status.applications:
                for app_name, app_info in status.applications.items():
                    route = getattr(app_info, 'route_prefix', 'N/A')
                    print(f"   - {app_name} (route: {route})")
                    if hasattr(app_info, 'deployments'):
                        for dep_name, dep_info in app_info.deployments.items():
                            dep_status = getattr(dep_info, 'status', 'unknown')
                            print(f"     └─ {dep_name}: {dep_status}")
            else:
                print("   No applications found.")
            return
        
        # Delete specific application
        app_name = args.app_name
        
        # Check if application exists
        if hasattr(status, 'applications') and app_name in status.applications:
            print(f"🗑️  Deleting application: {app_name}")
            try:
                serve.delete(app_name)
                print(f"✅ Successfully deleted application: {app_name}")
                print(f"   You can now redeploy it with your deployment script")
            except Exception as e:
                print(f"❌ Failed to delete '{app_name}': {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"⚠️  Application '{app_name}' not found")
            print("\n📊 Available applications:")
            if hasattr(status, 'applications') and status.applications:
                for existing_app in status.applications.keys():
                    print(f"   - {existing_app}")
            else:
                print("   No applications found.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
