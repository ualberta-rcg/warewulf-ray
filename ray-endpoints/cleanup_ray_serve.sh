#!/bin/bash
# Cleanup script for Ray Serve - stops Serve and cleans logs

# Don't use set -e here because we want to continue even if some cleanup steps fail
set +e

echo "🧹 Cleaning up Ray Serve..."

# Use the Ray Python from venv if available
if [ -f "/opt/ray/bin/python" ]; then
    PYTHON="/opt/ray/bin/python"
else
    PYTHON="python3"
fi

# Stop Ray Serve
echo "Stopping Ray Serve..."
$PYTHON << 'EOF'
from ray import serve
import ray

try:
    # Try to connect to Ray - get address from system
    import subprocess
    import socket
    
    # Get head node IP
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
    
    try:
        ray.init(address=ray_address, ignore_reinit_error=True)
    except Exception as e:
        print(f"⚠ Could not connect to Ray at {ray_address}: {e}")
        print("  Ray may not be running")
        exit(0)
    
    # Shutdown Serve
    try:
        # Check if Serve is running first
        serve.start(detached=True)
        # If we get here, Serve was running, so shut it down
        serve.shutdown()
        print("✓ Ray Serve stopped")
    except Exception as e:
        # Serve might not be running, that's fine
        print(f"⚠ Ray Serve may not be running: {e}")
        pass
except Exception as e:
    print(f"⚠ Error during cleanup: {e}")
EOF

# Clean up Ray Serve logs
echo "Cleaning Ray Serve logs..."
RAY_LOG_DIRS=(
    "/tmp/ray/session_*/logs/serve"
    "/var/log/ray/serve"
    "/tmp/ray/session_*/logs"
)

for pattern in "${RAY_LOG_DIRS[@]}"; do
    if ls $pattern 1> /dev/null 2>&1; then
        echo "  Cleaning: $pattern"
        find $pattern -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
        find $pattern -type f -name "*.out" -mtime +7 -delete 2>/dev/null || true
        find $pattern -type f -name "*.err" -mtime +7 -delete 2>/dev/null || true
    fi
done

# Clean up old session directories (older than 7 days)
if [ -d "/tmp/ray" ]; then
    echo "Cleaning old Ray session directories..."
    find /tmp/ray -maxdepth 1 -type d -name "session_*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true
fi

# Clean up metrics directories (keep recent, remove old)
METRICS_DIRS=(
    "/var/log/ray/sd-metrics"
    "/var/log/ray/sd-triton-metrics"
)

for dir in "${METRICS_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning metrics: $dir"
        find "$dir" -type f -name "*.jsonl" -mtime +30 -delete 2>/dev/null || true
    fi
done

echo "✅ Cleanup complete!"
echo ""
echo "Verifying Ray cluster status..."
$PYTHON << 'EOF'
import ray
import sys

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
    
    # Check Ray cluster status
    print("✓ Ray cluster is accessible")
    
    # Get cluster resources
    resources = ray.cluster_resources()
    print(f"  Cluster resources:")
    for key, value in sorted(resources.items()):
        if 'memory' in key.lower():
            print(f"    {key}: {value / (1024**3):.2f} GB")
        else:
            print(f"    {key}: {value}")
    
    # Check if Ray Serve is running
    from ray import serve
    try:
        serve.start(detached=True)
        status = serve.status()
        print(f"\n✓ Ray Serve status:")
        print(f"  Proxies: {len(status.proxies) if hasattr(status, 'proxies') else 0}")
        if hasattr(status, 'applications') and status.applications:
            print(f"  Applications: {len(status.applications)}")
            for app_name in status.applications.keys():
                print(f"    - {app_name}")
        else:
            print(f"  Applications: 0 (none deployed)")
    except Exception as e:
        print(f"  Ray Serve: Not running ({e})")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Error verifying Ray: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "To restart Ray Serve, run: bash quick_start.sh"
