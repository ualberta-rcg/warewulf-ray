#!/bin/bash
# Setup script for shared virtual environment on NFS
# This creates a venv in /data that can be used by all Ray workers

set -e

VENV_PATH="/data/ray-endpoints-venv"
PYTHON_VERSION="${1:-3.12}"

echo "🔧 Setting up shared virtual environment at $VENV_PATH"

# Check if /data is writable
if [ ! -w "/data" ]; then
    echo "❌ /data is not writable. Make sure you have permissions."
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python${PYTHON_VERSION} -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
else
    echo "⚠ Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        python${PYTHON_VERSION} -m venv "$VENV_PATH"
        echo "✓ Virtual environment recreated"
    else
        echo "Using existing virtual environment"
    fi
fi

# Activate and upgrade pip
echo "Upgrading pip..."
"$VENV_PATH/bin/pip" install --upgrade pip

# Install common packages
echo "Installing common packages..."
"$VENV_PATH/bin/pip" install \
    psutil \
    requests \
    numpy \
    onnxruntime \
    fastapi \
    uvicorn

# Optional: Install additional packages
echo ""
echo "Virtual environment ready at: $VENV_PATH"
echo ""
echo "To install additional packages:"
echo "  $VENV_PATH/bin/pip install package_name"
echo ""
echo "To use this venv in deployments, it will be automatically detected."
echo "The deployment code will add it to sys.path on worker nodes."
