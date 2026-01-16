# Ray Serve Triton Integration

Standalone model deployments using Triton Server, based on the [Ray Serve Triton integration guide](https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html).

## Structure

Each model is a standalone file that can be deployed individually:

- `stable_diffusion_triton.py` - Stable Diffusion with Triton Server
- `kandinsky3_triton.py` - Kandinsky3 with Triton Server

## Prerequisites

Install required packages in the Ray virtual environment:

```bash
# Install Pillow for image handling
/opt/ray/bin/pip install Pillow

# Install numpy if not already available
/opt/ray/bin/pip install numpy
```

## Usage

### Deploy Individual Models

Each file can be run directly using the Ray venv Python:

```bash
# Deploy Stable Diffusion
cd /data/models/warewulf-ray/ray-endpoints/endpoints
/opt/ray/bin/python stable_diffusion_triton.py

# Deploy Kandinsky3 (in another terminal or after stopping the first)
/opt/ray/bin/python kandinsky3_triton.py
```

**Important**: Always use `/opt/ray/bin/python` instead of system `python3` to ensure you're using the Ray virtual environment with all dependencies.

### Features

- **Scale-to-zero**: Models scale down when idle (`min_replicas=0`)
- **GPU support**: Uses `ray_actor_options={"num_gpus": 1}`
- **Triton Server**: Embedded Triton Server in each replica
- **Usage metrics**: Saved to `/var/log/ray/*-triton-metrics/`
- **Health checks**: `/health` endpoint for monitoring

## API Endpoints

### Stable Diffusion

- **Health**: `GET http://localhost:8000/api/v1/stable-diffusion-triton/health`
- **Generate**: `GET http://localhost:8000/api/v1/stable-diffusion-triton/generate?prompt=your+prompt`
- **Info**: `GET http://localhost:8000/api/v1/stable-diffusion-triton/info`

### Kandinsky3

- **Health**: `GET http://localhost:8000/api/v1/kandinsky3-triton/health`
- **Generate**: `GET http://localhost:8000/api/v1/kandinsky3-triton/generate?prompt=your+prompt`
- **Info**: `GET http://localhost:8000/api/v1/kandinsky3-triton/info`

## Model Repository Setup

Triton Server requires a model repository structure. For Stable Diffusion, you need:

```
model_repository/
├── stable_diffusion/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
├── text_encoder/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── vae/
    ├── 1/
    │   └── model.plan
    └── config.pbtxt
```

See the [Ray Serve Triton integration guide](https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html) for details on:
- Exporting models to ONNX/TensorRT
- Creating model repository structure
- Configuring `config.pbtxt` files

## Configuration

Update the `model_repository` path in each file to point to your actual model repository:

```python
model_repository = "/data/models/stablediffusion"  # Or "/data/ray-triton/model_repository" for Triton repo
```

You can also use remote storage like S3:
```python
model_repository = "s3://bucket-name/models"
```

## Notes

- Each deployment uses `@serve.ingress(app)` which automatically handles FastAPI integration
- Models are loaded on-demand when first requested
- Each replica runs its own Triton Server instance
- GPU resources are allocated per replica (`num_gpus: 1`)

## Triton Server Binary Location

When using PyTriton (embedded mode), the Triton Server binary is located at:
```
/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver
```

**Note**: The exact path depends on your Python version. For Python 3.12, it's in the path above. For other versions, adjust the `python3.12` part accordingly (e.g., `python3.11` for Python 3.11).

This binary is automatically used by PyTriton when you import `tritonserver` in Python. In `triton_inference.py`, we explicitly reference this path to start the server as a subprocess.

To verify the location on your system:
```bash
find /opt/ray -name "tritonserver" -type f 2>/dev/null
```

If the binary is not found, PyTriton may not be installed. Install it with:
```bash
/opt/ray/bin/pip install pytriton
```