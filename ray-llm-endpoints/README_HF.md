# HuggingFace LLM Endpoints

This directory contains deployment scripts for serving HuggingFace models using vLLM directly with Ray Serve.

## Features

- ✅ **No `ray.serve.llm` required** - Uses vLLM directly
- ✅ **HuggingFace token support** - For private/gated models
- ✅ **OpenAI-compatible API** - Works with standard OpenAI clients
- ✅ **Automatic worker installation** - Dependencies installed via `runtime_env`
- ✅ **NFS model caching** - Models cached on `/data/models` (shared NFS)

## Quick Start

### 1. Set HuggingFace Token (if needed)

```bash
export HF_TOKEN="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"
# Or
export HUGGING_FACE_HUB_TOKEN="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"
```

### 2. Deploy DeepSeek-7B Model

```bash
cd ray-llm-endpoints
bash deploy_hf.sh
```

Or with custom model:

```bash
MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat" bash deploy_hf.sh
```

### 3. Test the Endpoint

```bash
# Get your head node IP
HEAD_NODE_IP="172.26.92.232"  # Replace with your IP

# Test chat completion
curl http://${HEAD_NODE_IP}:9202/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai-deepseek-llm-7b-chat",
    "messages": [
      {"role": "user", "content": "Hello! What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Available Models

### DeepSeek-7B Chat
```bash
MODEL_NAME="deepseek-ai/deepseek-llm-7b-chat" bash deploy_hf.sh
```

### Other Models
Any HuggingFace model compatible with vLLM:
- `meta-llama/Llama-2-7b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `Qwen/Qwen-7B-Chat`
- etc.

## Configuration

### GPU Requirements

The deployment uses `num_gpus: 1` by default. Adjust in `serve_hf_models.py`:

```python
@serve.deployment(
    name="hf-llm-vllm",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1,  # Change this for multi-GPU
        # ...
    }
)
```

### Model Cache Location

Models are cached on `/data/models` (NFS share). To change:

```python
llm_kwargs = {
    "download_dir": "/data/models",  # Change this
    # ...
}
```

### Memory Configuration

Adjust `max_model_len` based on your GPU memory:

```python
llm_kwargs = {
    "max_model_len": 32768,  # Reduce if OOM errors
    # ...
}
```

## Troubleshooting

### "No space left on device"

1. Check disk space:
   ```bash
   df -h
   ```

2. Clean up pip cache:
   ```bash
   /opt/ray/bin/pip cache purge
   ```

3. Clean up Ray cache:
   ```bash
   rm -rf /tmp/ray-*
   ```

4. Use NFS for model cache (already configured):
   - Models are cached on `/data/models` (NFS share)

### "Failed to load model"

1. Check network access:
   ```bash
   curl -I https://huggingface.co
   ```

2. Verify HuggingFace token:
   ```bash
   echo $HF_TOKEN
   ```

3. Check model name is correct:
   - Visit https://huggingface.co/models
   - Use exact model ID (e.g., `deepseek-ai/deepseek-llm-7b-chat`)

### "vLLM not available"

vLLM is installed automatically on worker nodes via `runtime_env`. If deployment fails:

1. Check worker logs:
   ```bash
   ray logs
   ```

2. Manually install on head node (if needed):
   ```bash
   /opt/ray/bin/pip install vllm>=0.10.1
   ```
   (Note: Requires disk space)

### "Connection refused" on port 9202

1. Check Ray Serve is running:
   ```bash
   /opt/ray/bin/python -c "from ray import serve; print(serve.status())"
   ```

2. Restart Ray Serve:
   ```bash
   cd ../ray-endpoints
   bash restart_serve.sh
   ```

## API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```

### List Models
```
GET /v1/models
```

### Health Check
```
GET /v1/health
```

## Example Python Client

```python
import requests

HEAD_NODE_IP = "172.26.92.232"
url = f"http://{HEAD_NODE_IP}:9202/v1/chat/completions"

response = requests.post(url, json={
    "model": "deepseek-ai-deepseek-llm-7b-chat",
    "messages": [
        {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
})

print(response.json())
```

## Comparison with ray.serve.llm

| Feature | vLLM Direct (This) | ray.serve.llm |
|---------|---------------------|---------------|
| Installation | ✅ Automatic on workers | ❌ Requires `ray[serve-llm]` |
| Disk Space | ✅ Installed per-worker | ❌ Needs head node space |
| HuggingFace Token | ✅ Supported | ✅ Supported |
| OpenAI API | ✅ Compatible | ✅ Compatible |
| Streaming | ✅ Supported | ✅ Supported |

## Files

- `serve_hf_models.py` - Ray Serve deployment using vLLM
- `deploy_hf_models.py` - Deployment script
- `deploy_hf.sh` - Wrapper script with environment setup
