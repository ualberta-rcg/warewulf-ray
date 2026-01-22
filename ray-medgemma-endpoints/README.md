# MedGemma Model Deployment

Ray Serve deployment for MedGemma models with dynamic input discovery and autoscaling.

## Overview

MedGemma is a collection of Gemma 3 variants trained for medical text and image comprehension. This deployment supports:
- **Text-to-text**: Medical question answering and reasoning
- **Image-text-to-text**: Medical image analysis (X-rays, pathology, dermatology, ophthalmology)

## Features

- **Dynamic Input Discovery**: Automatically discovers what inputs each model accepts
- **Schema Endpoint**: Query `/schema` to see what parameters a model supports
- **Autoscaling**: Scale to zero when idle, scale up on demand
- **Compute Metrics**: Detailed resource usage in every response
- **Polling Support**: Returns 202 status when scaling up from zero
- **Error Reporting**: Loading errors are stored and returned via the API
- **Chat Format Support**: Supports both simple and chat message formats
- **Multimodal Support**: Handles both text-only and image+text inputs

## Model Information

- **Model**: `google/medgemma-27b-it` (27B parameter instruction-tuned multimodal model)
- **Size**: ~50GB+ (large model, requires significant GPU memory)
- **Requirements**: 
  - Transformers >= 4.50.0 (for Gemma 3 support)
  - HuggingFace token may be required (gated model - requires accepting terms)
  - Significant GPU memory (27B model)

## Quick Start

### Prerequisites

1. **Accept HuggingFace Terms**: Visit https://huggingface.co/google/medgemma-27b-it and accept the terms
2. **Set HuggingFace Token** (if required):
   ```bash
   export HF_TOKEN="your_token_here"
   ```

### Deploy the Model

**Default deployment (MedGemma 27B):**
```bash
./deploy_medgemma.sh
```

**Custom model path:**
```bash
./deploy_medgemma.sh --model-path google/medgemma-27b-it --app-name medgemma-27b
```

**Custom port:**
```bash
./deploy_medgemma.sh --port 9202
```

## API Endpoints

### Discover Model Schema

Get the dynamically discovered input schema:

```bash
curl http://<head-node-ip>:9202/medgemma-27b-it/v1/schema
```

**Response (when model is loaded):**
```json
{
  "model_id": "medgemma-27b-it",
  "model_class": "MedGemmaForConditionalGeneration",
  "processor_class": "MedGemmaProcessor",
  "capabilities": ["text-to-text", "image-text-to-text"],
  "inputs": {
    "required": ["messages"],
    "optional": ["max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty"]
  },
  "defaults": {
    "max_new_tokens": 200,
    "do_sample": false
  }
}
```

**Response (when model is loading):**
```json
{
  "error": "Model not loaded yet",
  "status": "loading"
}
```

### Generate Text (Text-Only)

Simple format:
```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "How do you differentiate bacterial from viral pneumonia?",
    "system": "You are a helpful medical assistant.",
    "max_new_tokens": 200
  }'
```

**Response:**
```json
{
  "status": "success",
  "model": "medgemma-27b-it",
  "response": "Bacterial and viral pneumonia can be differentiated through several clinical and diagnostic features...",
  "parameters": {
    "max_new_tokens": 200,
    "do_sample": false,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "repetition_penalty": null
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

### Generate Text (Image + Text)

For medical image analysis:
```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Describe this X-ray",
    "image": "data:image/png;base64,iVBORw0KGgoAAAANS...",
    "system": "You are an expert radiologist.",
    "max_new_tokens": 300
  }'
```

### Generate Text (Chat Format)

Using the full chat message format:
```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What are the symptoms of diabetes?"}
        ]
      }
    ],
    "max_new_tokens": 200
  }'
```

### Chat Format with Image

```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this X-ray"},
          {"type": "image", "image": "data:image/png;base64,..."}
        ]
      }
    ],
    "max_new_tokens": 300
  }'
```

### Health Check

```bash
curl http://<head-node-ip>:9202/medgemma-27b-it/v1/health
```

### List Models

```bash
curl http://<head-node-ip>:9202/medgemma-27b-it/v1/models
```

## Input Formats

### Simple Format

For basic use cases, you can use a simplified format:

```json
{
  "text": "Your question here",
  "system": "You are a helpful medical assistant.",
  "image": "data:image/png;base64,...",  // Optional
  "max_new_tokens": 200,
  "temperature": 0.7,  // Optional
  "top_p": 0.9,  // Optional
  "top_k": 50,  // Optional
  "repetition_penalty": 1.1  // Optional
}
```

### Chat Format

For more control, use the full chat message format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Your question"},
        {"type": "image", "image": "data:image/png;base64,..."}  // Optional
      ]
    }
  ],
  "max_new_tokens": 200,
  "temperature": 0.7
}
```

## Generation Parameters

- **max_new_tokens** (int, default: 200): Maximum number of tokens to generate
- **do_sample** (bool, default: false): Whether to use sampling
- **temperature** (float, optional): Sampling temperature (0.0-2.0)
- **top_p** (float, optional): Nucleus sampling parameter
- **top_k** (int, optional): Top-k sampling parameter
- **repetition_penalty** (float, optional): Repetition penalty (1.0 = no penalty)

## Scale-to-Zero Behavior

When scaled to zero:
- Returns HTTP 202 immediately with polling info
- No queuing - clients must retry/poll
- Model loads in background thread
- **Note**: 27B model takes 2-5 minutes to load, so retry after 60-120 seconds
- Once ready, requests process normally

## GPU Memory Considerations

The 27B MedGemma model is large and may require:
- **Single GPU**: At least 24GB+ VRAM (with bfloat16)
- **Multi-GPU**: Consider tensor parallelism for larger models
- **Quantization**: 4-bit/8-bit quantization can reduce memory (not implemented by default)

If you encounter OOM errors:
1. The model uses `device_map="auto"` which automatically handles device placement
2. Consider using a smaller model variant (4B) if available
3. Consider quantization (requires additional setup)

## Model Loading

The model is loaded with:
- **torch_dtype**: bfloat16 (falls back to float16 or default if needed)
- **device_map**: "auto" (automatically handles GPU/CPU placement)
- **trust_remote_code**: true (for custom model code)

## Compute Metrics

Each response includes `compute_metrics` with:
- Query duration
- Replica CPU usage
- Replica memory usage
- Replica GPU memory (allocated, reserved, total, used)
- Replica GPU utilization

Useful for accounting and monitoring resource usage per generation.

## Error Handling

If the model fails to load, the error is stored and returned via the API:

```json
{
  "error": "Model loading failed: <error message>",
  "status": "failed",
  "loading_error": "<detailed error message>"
}
```

Common issues:
- **Out of memory**: Model too large for GPU
- **HuggingFace token required**: Set `HF_TOKEN` environment variable
- **Terms not accepted**: Visit HuggingFace and accept model terms
- **Network issues**: Ensure access to huggingface.co

## Example Use Cases

### Medical Question Answering
```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What are the treatment options for type 2 diabetes?",
    "system": "You are a helpful medical assistant.",
    "max_new_tokens": 300
  }'
```

### X-Ray Analysis
```bash
# First, encode your X-ray image to base64
IMAGE_B64=$(base64 -w 0 xray.png)

curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"Describe this chest X-ray\",
    \"image\": \"data:image/png;base64,${IMAGE_B64}\",
    \"system\": \"You are an expert radiologist.\",
    \"max_new_tokens\": 400
  }"
```

### Pathology Image Analysis
```bash
curl -X POST http://<head-node-ip>:9202/medgemma-27b-it/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What do you observe in this histopathology slide?",
    "image": "data:image/png;base64,...",
    "system": "You are a pathologist.",
    "max_new_tokens": 300
  }'
```

## Troubleshooting

### Model takes too long to load
- 27B model is large (~50GB), first load can take 10-30 minutes
- Ensure fast network connection to HuggingFace
- Check disk space (need ~100GB+ free)

### Out of memory errors
- Model requires significant GPU memory
- Consider using smaller model variant if available
- Check GPU memory: `nvidia-smi`

### HuggingFace authentication errors
- Ensure you've accepted model terms on HuggingFace
- Set `HF_TOKEN` environment variable
- Verify token has access to gated models

### Transformers version issues
- Requires transformers >= 4.50.0
- Will be installed automatically via runtime_env
- If issues persist, manually install: `pip install transformers>=4.50.0`

## References

- [MedGemma on HuggingFace](https://huggingface.co/google/medgemma-27b-it)
- [MedGemma Technical Report](https://arxiv.org/abs/2507.05201)
- [Gemma 3 Documentation](https://huggingface.co/docs/transformers/model_doc/gemma3)
