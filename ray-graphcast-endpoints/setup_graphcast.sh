#!/bin/bash
# Setup script to clone GraphCast repository to NFS mount
# Run this once on the head node (or any node with access to /data)

set -e

GRAPHCAST_PATH="/data/models/graphcast"

echo "🔧 Setting up GraphCast repository..."
echo "   Target: ${GRAPHCAST_PATH}"

# Check if already cloned
if [ -d "${GRAPHCAST_PATH}/graphcast" ]; then
    echo "✅ GraphCast repository already exists at ${GRAPHCAST_PATH}"
    echo "   To update, run: cd ${GRAPHCAST_PATH} && git pull"
    exit 0
fi

# Create parent directory if needed
mkdir -p "$(dirname "${GRAPHCAST_PATH}")"

# Clone the repository
echo "📥 Cloning GraphCast from GitHub..."
if git clone https://github.com/google-deepmind/graphcast.git "${GRAPHCAST_PATH}"; then
    echo "✅ Successfully cloned GraphCast to ${GRAPHCAST_PATH}"
    echo ""
    echo "📦 Next steps:"
    echo "   1. Install JAX dependencies (if not already installed):"
    echo "      pip install 'jax[cuda12]' jaxlib dm-haiku dm-tree"
    echo "   2. Deploy GraphCast model:"
    echo "      bash deploy_graphcast.sh shermansiu/dm_graphcast"
else
    echo "❌ Failed to clone GraphCast repository"
    exit 1
fi
