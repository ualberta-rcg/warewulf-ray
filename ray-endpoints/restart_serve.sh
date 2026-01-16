#!/bin/bash
# Restart Ray Serve with HTTP options set to 0.0.0.0:8000
# This ensures Ray Serve is accessible from the network

set -e

echo "🔄 Restarting Ray Serve with network access (0.0.0.0:8001)..."
echo "⚠️  This will stop all currently deployed endpoints!"
echo "   You'll need to run ./deploy.sh again after this."
echo ""
# Auto-confirm if running non-interactively (for scripts)
if [ -t 0 ]; then
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
else
    echo "Non-interactive mode - proceeding automatically..."
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

# Start with HTTP options (port 8001 to avoid conflict with Triton on 8000)
RAY_SERVE_PORT = 8001
print(f"\nStarting Ray Serve with 0.0.0.0:{RAY_SERVE_PORT}...")
print("   (Triton is on port 8000, Ray Serve on port 8001)")
http_options = HTTPOptions(host="0.0.0.0", port=RAY_SERVE_PORT)
serve.start(detached=True, http_options=http_options)
print(f"✅ Ray Serve restarted on 0.0.0.0:{RAY_SERVE_PORT}")

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
echo "✅ Done! Ray Serve should now be accessible at http://<head-node-ip>:8001"
echo "   (Triton is on port 8000, Ray Serve is on port 8001)"
echo "   Now run: ./deploy.sh"
