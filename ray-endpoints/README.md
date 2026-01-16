# Ray Serve Endpoints with Triton Integration

This directory contains Ray Serve deployment code for serving models via NVIDIA Triton Inference Server.

## Directory Structure

```
ray-endpoints/
├── README.md                 # This file
├── requirements.txt          # Python dependencies for endpoints
├── deploy.py                # Main deployment script
├── deploy.sh                # Wrapper script (uses /opt/ray/bin/python)
├── endpoints/                # Individual endpoint deployments
│   ├── __init__.py
│   ├── onnx_models.py       # ONNX model endpoints (mobilenet, resnet, etc.)
│   └── stable_diffusion.py  # Stable Diffusion endpoint
├── config/                   # Configuration files
│   ├── triton_config.yaml   # Triton server configuration
│   └── ray_config.yaml      # Ray Serve configuration
└── utils/                    # Utility functions
    ├── __init__.py
    ├── triton_client.py     # Triton client wrapper
    └── model_loader.py      # Model loading utilities
```

## Setup

**Important**: Ray is installed in `/opt/ray` in the Docker container. You have two options:

### Option 1: Use Ray's Python directly (Recommended)
```bash
# Install dependencies in Ray's environment
/opt/ray/bin/pip install -r requirements.txt

# Deploy using Ray's Python
/opt/ray/bin/python deploy.py
```

### Option 2: Use wrapper script (Easiest)
```bash
# Make script executable (first time only)
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### Option 3: Use your virtual environment (for endpoint dependencies only)
```bash
# Activate the virtual environment for endpoint-specific dependencies
source /data/ray-endpoints-venv/bin/activate

# Install dependencies (Pillow, numpy, requests, etc.)
pip install -r requirements.txt

# Deploy using Ray's Python (Ray must come from /opt/ray)
/opt/ray/bin/python deploy.py
```

3. Ensure Ray cluster is running:
```bash
ray status
# or
/opt/ray/bin/ray status
```

## Model Locations

- **Triton Model Repository**: `/data/ray-triton/model_repository/`
- **Model Cache**: `/data/ray-triton/model_cache/`
- **Stable Diffusion Models**: `/data/models/stablediffusion/`
- **Triton Server Binary**: `/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver`

## Available Models

### ONNX Models (via Triton)
- `mobilenetv2`
- `resnet18`
- `resnet50`
- `shufflenet`
- `squeezenet`
- `vgg16`

### Stable Diffusion Models
- Various checkpoint and safetensors files in `/data/models/stablediffusion/`

## Deployment

Deploy all endpoints:
```bash
# Using wrapper script (recommended)
./deploy.sh

# Or using Ray's Python directly
/opt/ray/bin/python deploy.py
```

Deploy specific endpoint:
```bash
# Using wrapper script
./deploy.sh --endpoint onnx_models

# Or using Ray's Python directly
/opt/ray/bin/python deploy.py --endpoint onnx_models
```

## Endpoints

Once deployed, endpoints will be available at:
- Ray Serve Dashboard: `http://<head-node-ip>:8000`
- Individual endpoints: `http://<head-node-ip>:8000/<endpoint-name>`

## Configuration

Edit `config/triton_config.yaml` to configure Triton server settings.
Edit `config/ray_config.yaml` to configure Ray Serve settings.
