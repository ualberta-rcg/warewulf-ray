#!/bin/bash
# Restart Ray Serve with HTTP options set to 0.0.0.0:8000
# This ensures Ray Serve is accessible from the network

set -e

echo "🔄 Restarting Ray Serve with network access (0.0.0.0:8000)..."
echo "⚠️  This will stop all currently deployed endpoints!"
echo "   You'll need to run ./deploy.sh again after this."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Use Ray's Python
RAY_PYTHON="/opt/ray/bin/python"
if [ ! -f "$RAY_PYTHON" ]; then
    echo "❌ Ray Python not found at $RAY_PYTHON"
    exit 1
fi

# Python script to restart Serve
cat > /tmp/restart_serve.py << 'EOF'
import ray
from ray import serve
from ray.serve.config import HTTPOptions
import time

# Connect to Ray
ray.init(address="auto", ignore_reinit_error=True)

# Shutdown existing Serve
print("Stopping existing Ray Serve...")
try:
    serve.shutdown()
    print("✅ Stopped existing Ray Serve")
    time.sleep(2)  # Give it time to fully shutdown
except Exception as e:
    print(f"⚠️  Error stopping Serve: {e}")

# Start with HTTP options
print("\nStarting Ray Serve with 0.0.0.0:8000...")
http_options = HTTPOptions(host="0.0.0.0", port=8000)
serve.start(detached=True, http_options=http_options)
print("✅ Ray Serve restarted on 0.0.0.0:8000")

# Show status
time.sleep(1)
status = serve.status()
print(f"\n📊 Ray Serve Status:")
print(f"   Applications: {len(status.applications) if hasattr(status, 'applications') else 'N/A'}")
print(f"\n⚠️  Remember to redeploy your endpoints:")
print(f"   ./deploy.sh")
EOF

"$RAY_PYTHON" /tmp/restart_serve.py
rm -f /tmp/restart_serve.py

echo ""
echo "✅ Done! Ray Serve should now be accessible at http://<head-node-ip>:8000"
echo "   Now run: ./deploy.sh"
