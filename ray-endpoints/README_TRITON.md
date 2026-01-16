# Triton Inference Server Setup

This document explains how Triton Inference Server is configured in this project.

## Architecture Decision: Multi-Stage Build with Python 3.10

**We use a multi-stage build** to extract Triton from NVIDIA's official image, enabling both:
- ✅ **Triton server binary**: Standalone binary for process management
- ✅ **Triton Python API**: `import tritonserver` works (Python 3.10 compatible)

**Why Python 3.10?**
- Triton's Python bindings are compiled for Python 3.10
- Using Python 3.10 ensures `import tritonserver` works correctly
- Ray is installed with Python 3.10 to match Triton's version

## Triton Installation

Triton is installed via **multi-stage build** from NVIDIA's official image:

```dockerfile
# Multi-stage: Extract from NVIDIA image
FROM nvcr.io/nvidia/tritonserver:23.12-py3 AS triton-source

# Copy to main image
COPY --from=triton-source /opt/tritonserver /opt/tritonserver
```

**Binary location:**
- `/opt/tritonserver/bin/tritonserver` (from NVIDIA's official image)

**Python API:**
- `import tritonserver` works with Python 3.10 (matches Triton's Python version)
- Python bindings copied to `/opt/ray/lib/python3.10/site-packages/`

## Starting Triton Server

### Automatic Start (Recommended)

The `deploy.py` script automatically checks if Triton is running and starts it if needed:

```bash
cd ray-endpoints
bash deploy.sh
```

### Manual Start

If you need to start Triton manually:

```bash
cd ray-endpoints
bash start_triton.sh
```

### Check Triton Status

```bash
# Check if Triton is running
curl http://localhost:8000/v2/health/live

# Check Triton metrics
curl http://localhost:8002/metrics
```

## Configuration

### Ports

- **HTTP API**: Port 8000
- **gRPC API**: Port 8001 (Note: Ray Serve also uses 8001, but they don't conflict)
- **Metrics**: Port 8002

### Model Repository

Triton looks for models in:
```
/data/ray-triton/model_repository/
```

Model structure:
```
model_repository/
├── resnet50/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
├── mobilenetv2/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── ...
```

## Ray Serve Integration

Ray Serve endpoints use the **Triton HTTP client** to communicate with the standalone Triton server:

```python
import tritonclient.http as tritonhttp

client = tritonhttp.InferenceServerClient(url="localhost:8000")
```

This is configured in:
- `ray-endpoints/endpoints/onnx_models_simple.py`
- `ray-endpoints/endpoints/stable_diffusion.py`

## Why Not Embedded Triton?

We tried using the embedded Triton Python API (`import tritonserver`), but:

1. ❌ **Import issues**: The `tritonserver` module is not directly importable from `nvidia-pytriton` 0.7.0
2. ❌ **Complexity**: Embedded mode requires more configuration
3. ✅ **Standalone is simpler**: Separate process is easier to manage and debug

## Troubleshooting

### Triton Not Starting

1. Check if binary exists:
   ```bash
   ls -la /opt/tritonserver/bin/tritonserver
   # Or fallback location:
   ls -la /opt/ray/lib/python3.10/site-packages/pytriton/tritonserver/bin/tritonserver
   ```

2. Check if model repository exists:
   ```bash
   ls -la /data/ray-triton/model_repository/
   ```

3. Check port availability:
   ```bash
   netstat -tuln | grep 8000
   ```

### Connection Refused

If Ray Serve endpoints can't connect to Triton:

1. Verify Triton is running:
   ```bash
   curl http://localhost:8000/v2/health/live
   ```

2. Check Triton logs:
   ```bash
   # If started via start_triton.sh, check the process
   ps aux | grep tritonserver
   ```

3. Restart Triton:
   ```bash
   pkill tritonserver
   bash start_triton.sh
   ```

### Model Loading Issues

1. Check model repository structure:
   ```bash
   tree /data/ray-triton/model_repository/
   ```

2. Verify model files exist:
   ```bash
   find /data/ray-triton/model_repository -name "*.onnx"
   ```

3. Check Triton model status:
   ```bash
   curl http://localhost:8000/v2/repository/index
   ```

## Updating Triton

To update Triton, update `nvidia-pytriton` in the Dockerfile:

```dockerfile
/opt/ray/bin/pip install --no-cache-dir --upgrade "nvidia-pytriton"
```

Then rebuild the Docker image.

## Why Multi-Stage Build Instead of NVIDIA Base Image?

We use multi-stage build (extract from NVIDIA image) instead of using NVIDIA's image as base:

**Benefits:**
- ✅ Keep Ubuntu 24.04 base (better Warewulf compatibility)
- ✅ Full control over system configuration
- ✅ Can customize for HPC environments
- ✅ Still get Triton binary and Python API from official NVIDIA image

**What we extract:**
- Triton server binary (`/opt/tritonserver/bin/tritonserver`)
- Triton shared libraries (`/opt/tritonserver/lib/`)
- Triton Python bindings (for `import tritonserver`)
- Triton backends and configuration files

## References

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Ray Serve Triton Integration](https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html)
- [nvidia-pytriton Package](https://github.com/triton-inference-server/pytriton)
