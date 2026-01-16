# Stable Diffusion Endpoints

Ray Serve deployment for Stable Diffusion models using local `.safetensors` files.

## Features

- ✅ **Separate applications per model** - Each model gets its own app (no overwriting)
- ✅ **Autoscaling with scale-to-zero** - Scales down after 5 min idle, scales up on request
- ✅ **Polling support** - Returns 202 when scaled to zero, tells clients to retry
- ✅ **Compute metrics** - Shows GPU/CPU/memory usage per query for accounting
- ✅ **Local model files** - Uses `.safetensors` files from `/data/models/stablediffusion`
- ✅ **SDXL support** - Automatically detects and uses SDXL pipeline for XL models

## Quick Start

### 1. List Available Models

```bash
cd ray-stable-diffusion-endpoints
bash deploy_sd.sh --list
```

### 2. Deploy a Model

```bash
# Deploy SDXL base model
bash deploy_sd.sh /data/models/stablediffusion/sd_xl_base_1.0.safetensors

# Deploy SD 1.5 model
bash deploy_sd.sh /data/models/stablediffusion/v1-5-pruned-emaonly.safetensors
```

### 3. Generate Images

```bash
# Get your head node IP
HEAD_IP="172.26.92.232"

# Generate image
curl -X POST http://${HEAD_IP}:8000/sd-sd-xl-base-1-0/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a beautiful landscape with mountains and lakes",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }'
```

## API Endpoints

### Generate Image
```
POST /{app-name}/v1/generate
```

**Request:**
```json
{
  "prompt": "a beautiful landscape",
  "negative_prompt": "blurry, low quality",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "width": 512,
  "height": 512,
  "seed": 42,
  "format": "base64"
}
```

**Response:**
```json
{
  "status": "success",
  "model": "sd_xl_base_1.0",
  "image": "data:image/png;base64,...",
  "prompt": "a beautiful landscape",
  "parameters": {...},
  "compute_metrics": {
    "query_duration_seconds": 3.456,
    "replica_cpu_percent": 45.2,
    "replica_memory_mb": 12345.67,
    "replica_num_cpus": 8,
    "replica_gpu_memory_allocated_mb": 8192.0,
    "replica_gpu_memory_reserved_mb": 10240.0,
    "replica_gpu_memory_total_mb": 24576.0,
    "replica_gpu_utilization_percent": 85
  }
}
```

### Health Check
```
GET /{app-name}/v1/health
```

### List Models
```
GET /{app-name}/v1/models
```

## Model Detection

The deployment automatically detects:
- **SDXL models**: Files with "xl" or "sd_xl" in the name use `StableDiffusionXLPipeline`
- **Standard SD models**: All other models use `StableDiffusionPipeline`

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
