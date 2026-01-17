# Kandinsky Model Deployment

Ray Serve deployment for Kandinsky models with dynamic input discovery and autoscaling.

## Features

- **Dynamic Input Discovery**: Automatically discovers what inputs each model accepts
- **Schema Endpoint**: Query `/schema` to see what parameters a model supports
- **Autoscaling**: Scale to zero when idle, scale up on demand
- **Compute Metrics**: Detailed resource usage in every response
- **Polling Support**: Returns 202 status when scaling up from zero

## Quick Start

### List Available Models

```bash
./deploy_kandinsky.sh --list
```

### Deploy a Model

```bash
./deploy_kandinsky.sh /data/models/stablediffusion/kandinsky3-1.ckpt
```

Or using Python directly:

```bash
/opt/ray/bin/python deploy_kandinsky_v2.py \
  --model-path /data/models/stablediffusion/kandinsky3-1.ckpt \
  --model-name kandinsky3-1
```

## API Endpoints

### Discover Model Schema

Get the dynamically discovered input schema:

```bash
curl http://<head-node-ip>:8000/kandinsky-<model-name>/v1/schema
```

**Response:**
```json
{
  "model_id": "kandinsky3-1",
  "pipeline_class": "KandinskyV3Pipeline",
  "capabilities": ["text-to-image"],
  "inputs": {
    "required": ["prompt"],
    "optional": ["negative_prompt", "height", "width", "num_inference_steps", "guidance_scale", "generator", ...]
  },
  "methods": {
    "__call__": {
      "parameters": {
        "prompt": {
          "name": "prompt",
          "required": true,
          "default": null,
          "annotation": "Union[str, List[str]]"
        },
        "height": {
          "name": "height",
          "required": false,
          "default": 512,
          "annotation": "Optional[int]"
        },
        ...
      }
    }
  },
  "defaults": {
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512
  }
}
```

### Generate Image

```bash
curl -X POST http://<head-node-ip>:8000/kandinsky-<model-name>/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a beautiful landscape",
    "num_inference_steps": 50,
    "width": 512,
    "height": 512
  }'
```

The API automatically uses the discovered schema to:
- Validate inputs
- Apply defaults from the model
- Handle parameter types correctly (e.g., seed → generator)

**Response:**
```json
{
  "status": "success",
  "model": "kandinsky3-1",
  "image": "data:image/png;base64,...",
  "parameters": {
    "prompt": "a beautiful landscape",
    "num_inference_steps": 50,
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5
  },
  "compute_metrics": {
    "query_duration_seconds": 3.456,
    "replica_cpu_percent": 45.2,
    "replica_memory_mb": 12345.67,
    "replica_gpu_memory_allocated_mb": 8192.0,
    "replica_gpu_utilization_percent": 85
  }
}
```

### Health Check

```bash
curl http://<head-node-ip>:8000/kandinsky-<model-name>/v1/health
```

### List Models

```bash
curl http://<head-node-ip>:8000/kandinsky-<model-name>/v1/models
```

## How Dynamic Input Discovery Works

1. **Model Loading**: When the model loads, the pipeline's `__call__` method signature is inspected using Python's `inspect` module
2. **Schema Discovery**: Required vs optional parameters are identified, along with their types and defaults
3. **Capability Detection**: Available methods (like `image_to_image`, `inpaint`) are detected to determine capabilities
4. **Dynamic API**: The `/generate` endpoint uses the discovered schema to:
   - Accept only valid parameters
   - Apply defaults from the model
   - Handle type conversions (e.g., seed → generator, base64 → PIL Image)

## Model Detection

The deployment automatically detects:
- **Kandinsky V3**: Files with "kandinsky3" or "v3" in the name use `KandinskyV3Pipeline`
- **Kandinsky V2.2**: All other Kandinsky models use `KandinskyV22Pipeline`
- **File Format**: Supports both `.ckpt` and `.safetensors` files

## Scale-to-Zero Behavior

When scaled to zero:
- Returns HTTP 202 immediately with polling info
- No queuing - clients must retry/poll
- Model loads in background thread
- Once ready, requests process normally

## Compute Metrics

Each response includes `compute_metrics` with:
- Query duration
- Replica CPU usage
- Replica memory usage
- Replica GPU memory (allocated, reserved, total, used)
- Replica GPU utilization

Useful for accounting and monitoring resource usage per generation.
