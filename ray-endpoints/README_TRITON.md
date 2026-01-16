# Triton Inference Server Setup

This document explains how Triton Inference Server is configured in this project.

## Architecture Decision: Standalone Triton Binary

**We use the standalone Triton binary** (not the Python API) for the following reasons:

1. ✅ **Better for Warewulf**: Standalone binary is more reliable in containerized HPC environments
2. ✅ **More control**: Separate process allows better resource management
3. ✅ **Ubuntu 24.04 compatibility**: Works well with our base image
4. ✅ **Already available**: Binary comes with `nvidia-pytriton` package

## Triton Binary Location

The Triton server binary is installed via `nvidia-pytriton` in the Dockerfile:

```dockerfile
/opt/ray/bin/pip install --no-cache-dir "nvidia-pytriton"
```

Binary location:
```
/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver
```

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
   ls -la /opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver
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

## Alternative: Using NVIDIA Base Image

If you want to use the NVIDIA Triton base image instead:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:23.12-py3
# ... rest of dockerfile
```

However, this is **not recommended** for Warewulf because:
- Less control over system configuration
- May conflict with Warewulf-specific setup
- Harder to customize for HPC environments

## References

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Ray Serve Triton Integration](https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html)
- [nvidia-pytriton Package](https://github.com/triton-inference-server/pytriton)
