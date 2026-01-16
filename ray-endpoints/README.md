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
# Note: Dependencies will also be installed on workers via runtime_env
/opt/ray/bin/pip install -r requirements.txt

# Deploy using Ray's Python
/opt/ray/bin/python deploy.py
```

**Note:** The deployments use Ray runtime environments to automatically install dependencies (Pillow, numpy) on all worker nodes. This ensures workers have the required packages even if they're not installed globally.

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

3. **Start Triton Inference Server** (required before deploying endpoints):
```bash
# Start Triton server (run in background or separate terminal)
chmod +x start_triton.sh
./start_triton.sh

# Or run in background:
nohup ./start_triton.sh > triton.log 2>&1 &

# Check if Triton is running:
curl http://localhost:8000/v2/health/live
```

4. Ensure Ray cluster is running:
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

**Prerequisites:**
1. Triton server must be running (see Setup step 3)
2. Ray cluster must be running
3. Dependencies installed in Ray environment

Deploy all endpoints:
```bash
# Using wrapper script (recommended - installs deps automatically)
./deploy.sh

# Or using Ray's Python directly
/opt/ray/bin/python deploy.py
```

**Note:** If you see "Triton server is not live" errors, make sure Triton is running:
```bash
# Check Triton status
curl http://localhost:8000/v2/health/live

# If not running, start it:
./start_triton.sh
```

**Note:** Ray Serve is configured to listen on `0.0.0.0:8000` for network access. If you need to restart Ray Serve with these settings:
```bash
chmod +x restart_serve.sh
./restart_serve.sh
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
- Ray Serve Dashboard: `http://<head-node-ip>:8265`
- Ray Serve API: `http://<head-node-ip>:8000`

### Available Endpoints

1. **Proxy Endpoint** (`/onnx`): Use any model via query parameter
   ```bash
   curl -X POST "http://localhost:8000/onnx?model=resnet50" \
        -H "Content-Type: image/jpeg" \
        --data-binary @image.jpg
   ```

2. **ResNet50 Endpoint** (`/resnet50`): Dedicated ResNet50 endpoint
   ```bash
   curl -X POST "http://localhost:8000/resnet50" \
        -H "Content-Type: image/jpeg" \
        --data-binary @image.jpg
   ```

3. **MobileNetV2 Endpoint** (`/mobilenetv2`): Dedicated MobileNetV2 endpoint
   ```bash
   curl -X POST "http://localhost:8000/mobilenetv2" \
        -H "Content-Type: image/jpeg" \
        --data-binary @image.jpg
   ```

### Testing with curl

Use the provided test script:
```bash
chmod +x curl_examples.sh
./curl_examples.sh
```

Or test manually:
```bash
# Test ResNet50
curl -X POST "http://localhost:8000/resnet50" \
     -H "Content-Type: image/jpeg" \
     --data-binary @test_image.jpg

# Test with proxy endpoint
curl -X POST "http://localhost:8000/onnx?model=mobilenetv2" \
     -H "Content-Type: image/jpeg" \
     --data-binary @test_image.jpg
```

### Response Format

Successful inference returns:
```json
{
  "model": "resnet50",
  "predicted_class": 281,
  "confidence": 0.95,
  "predictions": [0.95, 0.02, 0.01, ...]
}
```

## Checking Routes and Endpoints

To verify that your endpoints are deployed and accessible:

```bash
# Use the check script
chmod +x check_routes.sh
./check_routes.sh

# Or check manually
ray serve status

# Or check via dashboard API
curl http://<head-node-ip>:8265/api/serve/applications/
```

The check script will:
- Show Ray Serve status
- Query the dashboard API for deployed applications
- Test each endpoint to see if it's accessible
- Show what's listening on port 8000

## Configuration

Edit `config/triton_config.yaml` to configure Triton server settings.
Edit `config/ray_config.yaml` to configure Ray Serve settings.

**Note:** Ray Serve is configured to listen on `0.0.0.0:8000` so it's accessible from the network, not just localhost.
