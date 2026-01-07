#!/bin/bash
# Install dependencies for Triton model endpoints

echo "📦 Installing dependencies for Triton model endpoints..."

# Use Ray venv pip
PIP="/opt/ray/bin/pip"

if [ ! -f "$PIP" ]; then
    echo "❌ Ray venv not found at /opt/ray/bin/pip"
    echo "   Make sure Ray is installed in /opt/ray"
    exit 1
fi

echo "Installing Pillow..."
$PIP install Pillow

echo "Installing numpy..."
$PIP install numpy

echo "Checking for tritonserver..."
$PIP list | grep -i triton || echo "⚠ tritonserver not found - install nvidia-pytriton if needed"

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "You can now run:"
echo "  /opt/ray/bin/python endpoints/stable_diffusion_triton.py"
echo "  /opt/ray/bin/python endpoints/kandinsky3_triton.py"
