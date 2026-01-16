#!/bin/bash
# Verify that ray-endpoints is accessible on NFS at /data/warewulf-ray

echo "🔍 Verifying NFS setup for ray-endpoints..."
echo ""

# Check if /data exists
if [ ! -d "/data" ]; then
    echo "❌ /data directory does not exist"
    echo "   Make sure /data is mounted (NFS share)"
    exit 1
fi

echo "✓ /data directory exists"

# Check for ray-endpoints in common locations
FOUND=false
POSSIBLE_PATHS=(
    "/data/warewulf-ray/warewulf-ray/ray-endpoints"
    "/data/warewulf-ray/ray-endpoints"
    "/data/ray-endpoints"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/base.py" ]; then
        echo "✓ Found ray-endpoints at: $path"
        echo "  base.py exists: $(test -f "$path/base.py" && echo 'yes' || echo 'no')"
        echo "  endpoints/ exists: $(test -d "$path/endpoints" && echo 'yes' || echo 'no')"
        FOUND=true
        break
    fi
done

if [ "$FOUND" = false ]; then
    echo "⚠ ray-endpoints not found on /data"
    echo ""
    echo "To fix this:"
    echo "  1. Clone the repo to /data:"
    echo "     cd /data"
    echo "     git clone <your-repo-url> warewulf-ray"
    echo ""
    echo "  2. Or create a symlink:"
    echo "     ln -s /root/warewulf-ray /data/warewulf-ray"
    echo ""
    echo "  3. Verify base.py is accessible:"
    echo "     ls -la /data/warewulf-ray/warewulf-ray/ray-endpoints/base.py"
    exit 1
fi

# Check if shared venv exists
if [ -d "/data/ray-endpoints-venv" ]; then
    echo "✓ Shared venv found at: /data/ray-endpoints-venv"
    if [ -f "/data/ray-endpoints-venv/bin/python" ]; then
        echo "  Python: /data/ray-endpoints-venv/bin/python"
    fi
else
    echo "⚠ Shared venv not found (optional)"
    echo "  Create with: bash setup_shared_venv.sh"
fi

echo ""
echo "✅ NFS setup looks good!"
echo ""
echo "The deployment will use: $path"
echo "Workers should be able to access base.py and endpoint modules from this location."
