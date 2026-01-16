#!/bin/bash
# Install dependencies for model endpoints
# Uses shared venv at /data/ray-endpoints-venv if available, otherwise /opt/ray

echo "📦 Installing dependencies for model endpoints..."

# Check for shared venv first (preferred for NFS setups)
SHARED_VENV="/data/ray-endpoints-venv"
if [ -d "$SHARED_VENV" ] && [ -f "$SHARED_VENV/bin/pip" ]; then
    PIP="$SHARED_VENV/bin/pip"
    PYTHON="$SHARED_VENV/bin/python"
    echo "✓ Using shared venv: $SHARED_VENV"
else
    # Fall back to Ray venv
    PIP="/opt/ray/bin/pip"
    PYTHON="/opt/ray/bin/python"
    if [ ! -f "$PIP" ]; then
        echo "❌ Neither shared venv nor Ray venv found"
        echo "   Shared venv: $SHARED_VENV (not found)"
        echo "   Ray venv: /opt/ray/bin/pip (not found)"
        echo ""
        echo "   To create shared venv:"
        echo "     bash setup_shared_venv.sh"
        exit 1
    fi
    echo "⚠ Using Ray venv: /opt/ray (shared venv not found)"
    echo "   Consider creating shared venv: bash setup_shared_venv.sh"
fi

echo ""
echo "Installing Pillow..."
$PIP install Pillow

echo "Installing numpy..."
$PIP install numpy

echo "Installing diffusers and torch (for direct model loading)..."
$PIP install diffusers torch torchvision

echo "Installing fastapi and uvicorn (if not already installed)..."
$PIP install fastapi uvicorn

echo "Checking for tritonserver..."
$PIP list | grep -i triton || echo "⚠ tritonserver not found - will use diffusers fallback"

echo ""
echo "✅ Dependencies installed in: $(dirname $PIP)"
echo ""
echo "To verify installation:"
echo "  $PYTHON -c 'import diffusers, torch, fastapi; print(\"✓ All packages available\")'"
echo ""
echo "You can now deploy endpoints:"
echo "  bash quick_start.sh"
