#!/bin/bash
# Check what Ray packages are installed

echo "🔍 Checking Ray Installation"
echo "============================"
echo ""

RAY_PYTHON="/opt/ray/bin/python"

if [ ! -f "$RAY_PYTHON" ]; then
    echo "❌ Ray Python not found at $RAY_PYTHON"
    exit 1
fi

echo "📦 Ray Version:"
$RAY_PYTHON -c "import ray; print(f'  Ray: {ray.__version__}')" 2>/dev/null || echo "  ❌ Ray not importable"

echo ""
echo "📦 Installed Ray Packages:"
$RAY_PYTHON -m pip list | grep -i ray || echo "  (no ray packages found)"

echo ""
echo "🔍 Checking Ray Extras:"
echo "  - ray[default]:"
$RAY_PYTHON -c "import ray" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo "  - ray[serve]:"
$RAY_PYTHON -c "from ray import serve" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo "  - ray.serve.llm:"
$RAY_PYTHON -c "from ray.serve.llm import LLMConfig, build_openai_app" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo ""
echo "📊 Ray Installation Info:"
$RAY_PYTHON -c "
import ray
import sys
print(f'  Python: {sys.executable}')
print(f'  Ray version: {ray.__version__}')
print(f'  Ray path: {ray.__file__}')
try:
    import ray.serve
    print(f'  Ray Serve path: {ray.serve.__file__}')
except:
    print('  Ray Serve: Not available')
try:
    from ray.serve import llm
    print(f'  Ray Serve LLM path: {llm.__file__}')
except ImportError as e:
    print(f'  Ray Serve LLM: Not available ({e})')
"

echo ""
echo "🔍 Checking Triton Installation:"
echo "================================"
echo ""

echo "📦 Installed Triton Packages:"
$RAY_PYTHON -m pip list | grep -i triton || echo "  (no triton packages found)"

echo ""
echo "  - tritonclient (HTTP/gRPC client):"
$RAY_PYTHON -c "import tritonclient.http as tritonhttp" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo "  - tritonclient.grpc:"
$RAY_PYTHON -c "import tritonclient.grpc as tritongrpc" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo "  - nvidia-pytriton (Python API):"
$RAY_PYTHON -c "import tritonserver" 2>/dev/null && echo "    ✅ Available" || echo "    ❌ Not available"

echo ""
echo "📊 Triton Installation Info:"
$RAY_PYTHON -c "
try:
    import tritonclient.http as tritonhttp
    print(f'  tritonclient.http: {tritonhttp.__file__}')
except ImportError as e:
    print(f'  tritonclient.http: Not available ({e})')

try:
    import tritonserver
    print(f'  tritonserver (nvidia-pytriton): {tritonserver.__file__}')
    # Check if tritonserver binary is available
    import os
    # Try NVIDIA's Triton binary first, then nvidia-pytriton
    triton_bin = '/opt/tritonserver/bin/tritonserver'
    if not os.path.exists(triton_bin):
        triton_bin = '/opt/ray/lib/python3.10/site-packages/pytriton/tritonserver/bin/tritonserver'
    if os.path.exists(triton_bin):
        print(f'  Triton server binary: {triton_bin} ✅')
    else:
        print(f'  Triton server binary: Not found at expected location')
except ImportError as e:
    print(f'  tritonserver: Not available ({e})')
"

echo ""
echo "💡 To check what ray[all] includes:"
echo "   $RAY_PYTHON -m pip show ray | grep -A 20 'Provides-Extra'"
echo ""
echo "💡 To check Triton package details:"
echo "   $RAY_PYTHON -m pip show tritonclient nvidia-pytriton"
