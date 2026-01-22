# Ray Serve LLM Endpoints

This directory contains Ray Serve LLM deployment for various LLM models.

## Quick Start

### Option 1: HuggingFace Models (Recommended - No ray.serve.llm needed)

For HuggingFace models like DeepSeek-7B, use the vLLM direct approach:

```bash
export HF_TOKEN="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"
cd ray-llm-endpoints
bash deploy_hf.sh
```

See [README_HF.md](README_HF.md) for details.

### Option 2: GPT-OSS Models (vLLM Direct)

For GPT-OSS models without ray.serve.llm:

```bash
cd ray-llm-endpoints
/opt/ray/bin/python deploy_vllm.py --model openai-community/gpt-oss-20b
```

### Option 3: GPT-OSS Models (ray.serve.llm)

For GPT-OSS models using Ray Serve LLM extension (requires `ray[serve-llm]`):

```bash
cd ray-llm-endpoints
bash deploy.sh
```

**Note:** `ray[serve-llm]` is not included in `ray[all]` and requires additional installation.

---

## GPT-OSS Deployment (Original)

Based on: [Ray Serve LLM GPT-OSS Tutorial](https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm/gpt-oss/README.html)

## Directory Structure

```
ray-llm-endpoints/
├── README.md              # This file
├── requirements.txt       # Python dependencies (vllm, etc.)
├── deploy.py             # Main deployment script
├── deploy.sh             # Wrapper script (uses /opt/ray/bin/python)
├── serve_gpt_oss.py      # GPT-OSS deployment configuration
└── config/               # Configuration files
    └── llm_config.yaml   # LLM-specific configuration
```

## Prerequisites

1. **Ray cluster must be running**
2. **vLLM installed** (will be installed via requirements.txt)
3. **GPU support** (recommended for LLM inference)

## Installation

```bash
# Install dependencies in Ray's environment
/opt/ray/bin/pip install -r requirements.txt
```

## Available Models

- `gpt-oss-20b` - 20 billion parameter model
- `gpt-oss-120b` - 120 billion parameter model

## Deployment

### Option 1: Using vLLM Directly (Recommended - No ray.serve.llm needed)

This approach uses vLLM directly with Ray Serve, avoiding the need for `ray.serve.llm`:

```bash
# Deploy GPT-OSS endpoint using vLLM directly
/opt/ray/bin/python deploy_vllm.py

# Or deploy specific model
/opt/ray/bin/python deploy_vllm.py --model openai-community/gpt-oss-20b
```

### Option 2: Using Ray Serve LLM (Requires ray.serve.llm)

If you have `ray.serve.llm` available (Ray 2.49.0+ with serve-llm extension):

```bash
# Deploy GPT-OSS endpoint
./deploy.sh

# Or deploy specific model
./deploy.sh --model gpt-oss-20b
```

**Note:** If you get "ray.serve.llm not available", use Option 1 instead.

### Manual Deployment

```bash
# Using Ray's Python directly
/opt/ray/bin/python deploy.py

# Deploy specific model
/opt/ray/bin/python deploy.py --model gpt-oss-20b
```

## Usage

### OpenAI-Compatible API

The endpoint provides an OpenAI-compatible API at:
- **Base URL**: `http://<head-node-ip>:9200/v1`
- **Model name**: `my-gpt-oss` (or as configured)

### Example: Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://172.26.92.232:9200/v1",
    api_key="not-needed"  # API key not required for local deployment
)

response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        {"role": "user", "content": "How many r's in strawberry?"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Example: cURL

```bash
curl http://172.26.92.232:9200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-gpt-oss",
    "messages": [
      {"role": "user", "content": "How many r's in strawberry?"}
    ],
    "stream": false
  }'
```

### Reasoning Configuration

GPT-OSS supports reasoning with three effort levels: `low`, `medium`, `high` (default: `medium`)

```python
response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        {"role": "user", "content": "What are the three main touristic spots in Paris?"}
    ],
    reasoning_effort="low"  # or "medium", "high"
)
```

## Configuration

Edit `serve_gpt_oss.py` to configure:
- Model name/path
- GPU resources
- Max concurrency
- KV cache settings
- Reasoning effort

## Monitoring

- **Ray Dashboard**: `http://<head-node-ip>:8265`
- **Serve LLM Dashboard**: Available in Ray Dashboard under Metrics
- **Enable LLM metrics**: Set `log_engine_metrics: true` in config

## Troubleshooting

### vLLM Installation Issues

If vLLM fails to install, you may need build tools:
```bash
sudo apt-get update
sudo apt-get install -y build-essential
/opt/ray/bin/pip install vllm>=0.10.1
```

### GPU Not Detected

Ensure:
- CUDA is available
- GPU devices are accessible
- Ray can detect GPUs: `ray status`

### Model Download Issues

Models are downloaded from HuggingFace. Ensure:
- Network access to `huggingface.co`
- Sufficient disk space for model weights
- Proper permissions in model cache directory

## Port Configuration

- **Ray Serve HTTP**: Port 9200 (matches other endpoints: kandinsky=9201, medgemma=9202)
- **Ray Dashboard**: Port 8265
- **OpenAI API**: `http://<head-node-ip>:9200/v1`

## Shutdown

```bash
# Shutdown Ray Serve LLM
/opt/ray/bin/python -c "from ray import serve; import ray; ray.init(address='auto'); serve.shutdown()"
```
