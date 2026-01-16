# Ray Serve Endpoints with Triton Integration

This directory contains Ray Serve deployment code for serving models via NVIDIA Triton Inference Server.

## Directory Structure

```
ray-endpoints/
├── README.md                 # This file
├── requirements.txt          # Python dependencies for endpoints
├── deploy.py                # Main deployment script
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

1. Activate the virtual environment:
```bash
source /data/ray-endpoints-venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ray cluster is running:
```bash
ray status
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
python deploy.py
```

Deploy specific endpoint:
```bash
python deploy.py --endpoint onnx_models
```

## Endpoints

Once deployed, endpoints will be available at:
- Ray Serve Dashboard: `http://<head-node-ip>:8000`
- Individual endpoints: `http://<head-node-ip>:8000/<endpoint-name>`

## Configuration

Edit `config/triton_config.yaml` to configure Triton server settings.
Edit `config/ray_config.yaml` to configure Ray Serve settings.
