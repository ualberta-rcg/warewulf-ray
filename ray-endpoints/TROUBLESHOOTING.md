# Ray Serve Troubleshooting Guide

## Issue: "TritonInferenceDeployment is unavailable" Error

If you see an error like:
```
Deployment(name='TritonInferenceDeployment', app='triton-inference') is unavailable because it failed to deploy.
```

This indicates a stale/failed deployment in Ray Serve. Here's how to fix it:

### Quick Fix

1. **Clean up stale deployments:**
   ```bash
   cd ~/warewulf-ray/ray-endpoints
   /opt/ray/bin/python cleanup_serve.py
   ```

2. **Redeploy endpoints:**
   ```bash
   bash quick_start.sh
   ```

### Manual Cleanup

If the cleanup script doesn't work, you can manually clean up:

```python
/opt/ray/bin/python << 'EOF'
from ray import serve
import ray

ray.init(address='auto', ignore_reinit_error=True)
serve.start(detached=True)

# Try to delete the problematic deployment/app
try:
    serve.delete("TritonInferenceDeployment")
    print("✓ Deleted TritonInferenceDeployment")
except Exception as e:
    print(f"Could not delete deployment: {e}")

try:
    serve.delete("triton-inference")
    print("✓ Deleted triton-inference app")
except Exception as e:
    print(f"Could not delete app: {e}")
EOF
```

### Nuclear Option: Restart Ray Serve

If nothing else works, restart Ray Serve completely:

```python
/opt/ray/bin/python << 'EOF'
from ray import serve
import ray

ray.init(address='auto', ignore_reinit_error=True)
serve.shutdown()
print("✓ Ray Serve shutdown")
serve.start(detached=True)
print("✓ Ray Serve restarted")
EOF
```

Then redeploy:
```bash
bash quick_start.sh
```

## Network Access: Making Endpoints Accessible on 0.0.0.0

The deployment script has been updated to automatically configure Ray Serve to listen on `0.0.0.0:8000` instead of just `localhost:8000`. This makes the endpoints accessible from other machines on the network.

### Verify Network Access

After deployment, check that Ray Serve is listening on all interfaces:

```bash
netstat -tlnp | grep 8000
# or
ss -tlnp | grep 8000
```

You should see something like:
```
0.0.0.0:8000    # Listening on all interfaces
```

If you see `127.0.0.1:8000` instead, Ray Serve is only listening on localhost.

### Access Endpoints from Network

Once configured, you can access endpoints from other machines:

```bash
# From another machine
curl http://<head-node-ip>:8000/api/v1/stable-diffusion/health
curl http://<head-node-ip>:8000/api/v1/kandinsky3/health
```

Replace `<head-node-ip>` with the actual IP address of your Ray head node.

### Firewall Configuration

Make sure port 8000 is open in your firewall:

```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

## Checking Ray Serve Status

To see what's currently deployed:

```python
/opt/ray/bin/python << 'EOF'
from ray import serve
import ray

ray.init(address='auto', ignore_reinit_error=True)
serve.start(detached=True)

status = serve.status()
print("=== Ray Serve Status ===")
print(f"HTTP options: {status.http_options}")

if hasattr(status, 'applications') and status.applications:
    print("\n=== Applications ===")
    for app_name, app_info in status.applications.items():
        print(f"\nApp: {app_name}")
        if hasattr(app_info, 'status'):
            print(f"  Status: {app_info.status}")
        if hasattr(app_info, 'deployments'):
            for dep_name, dep_info in app_info.deployments.items():
                print(f"  Deployment: {dep_name}")
                if hasattr(dep_info, 'status'):
                    print(f"    Status: {dep_info.status}")
                if hasattr(dep_info, 'replicas'):
                    print(f"    Replicas: {dep_info.replicas}")
EOF
```

## Common Issues

### Endpoints return 503 or "unavailable"

- **Cause**: Deployment scaled to zero (MIN_REPLICAS=0) and hasn't scaled up yet
- **Solution**: Wait a few seconds and try again, or send a request to trigger scale-up

### Can't access from network

- **Cause**: Ray Serve only listening on localhost
- **Solution**: The updated `deploy.py` should fix this automatically. If not, manually configure:
  ```python
  from ray.serve.config import HTTPOptions
  serve.start(detached=True, http_options=HTTPOptions(host="0.0.0.0", port=8000))
  ```

### Deployment fails to start

- **Cause**: Missing dependencies, model files, or GPU issues
- **Solution**: Check logs with `ray logs` or check the Ray dashboard at `http://<head-node-ip>:8265`
