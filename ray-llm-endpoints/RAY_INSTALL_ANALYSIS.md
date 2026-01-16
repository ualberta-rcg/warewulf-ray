# Ray Installation Analysis

## What's Currently Installed in Dockerfile

Looking at `dockerfile` lines 156-168:

```dockerfile
RUN python3 -m venv /opt/ray && \
    /opt/ray/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/ray/bin/pip install --no-cache-dir "ray[all]" && \
    /opt/ray/bin/pip install --no-cache-dir "tritonclient[all]" && \
    /opt/ray/bin/pip install --no-cache-dir "nvidia-pytriton" && \
    ...
```

### Currently Installed:

1. **`ray[all]`** - Includes:
   - `default` - Core Ray functionality
   - `serve` - Ray Serve for model serving
   - `tune` - Ray Tune for hyperparameter tuning
   - `rllib` - Ray RLlib for reinforcement learning
   - `data` - Ray Data for data processing
   - `train` - Ray Train for distributed training
   - `air` - Ray AIR for end-to-end ML workflows
   - `gpu` - GPU support and CUDA integration
   - And other optional dependencies

2. **`tritonclient[all]`** - Triton client libraries (HTTP, gRPC)

3. **`nvidia-pytriton`** - PyTriton wrapper for embedded Triton

### What's MISSING:

❌ **`ray[serve,llm]` or `ray[llm]`** - LLM serving components
   - This includes `ray.serve.llm` module
   - Provides `LLMConfig`, `build_openai_app`, etc.
   - Required for the official Ray Serve LLM API

## Why `ray.serve.llm` is Not Available

According to Ray documentation:
- `ray[all]` does **NOT** include LLM serving components
- You need to explicitly install `ray[serve,llm]` or `ray[llm]` for LLM support
- The LLM extras are separate from `ray[all]`

## Solutions

### Option 1: Add LLM Support to Dockerfile (Recommended for Future)

Modify the dockerfile to include:

```dockerfile
/opt/ray/bin/pip install --no-cache-dir "ray[all]" && \
/opt/ray/bin/pip install --no-cache-dir "ray[serve,llm]" && \
```

Or replace `ray[all]` with:

```dockerfile
/opt/ray/bin/pip install --no-cache-dir "ray[all,serve,llm]" && \
```

### Option 2: Use vLLM Direct (Current Workaround)

The `serve_gpt_oss_vllm.py` implementation uses vLLM directly, which:
- ✅ Works without `ray.serve.llm`
- ✅ Provides OpenAI-compatible API
- ✅ Uses vLLM's `LLM` class directly
- ✅ Installs vLLM via `runtime_env` on workers

### Option 3: Install LLM Support at Runtime

If you can't modify the dockerfile, install at runtime:

```bash
/opt/ray/bin/pip install "ray[serve,llm]"
```

**Note:** This requires disk space, which you're currently out of.

## Check What's Actually Installed

Run this in your container:

```bash
/opt/ray/bin/python -c "
import ray
print(f'Ray version: {ray.__version__}')

# Check what's available
try:
    from ray import serve
    print('✅ ray.serve available')
except:
    print('❌ ray.serve NOT available')

try:
    from ray.serve.llm import LLMConfig, build_openai_app
    print('✅ ray.serve.llm available')
except ImportError as e:
    print(f'❌ ray.serve.llm NOT available: {e}')
"
```

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| `ray[all]` | ✅ Installed | Core Ray + Serve + Tune + etc. |
| `ray.serve` | ✅ Available | Part of `ray[all]` |
| `ray.serve.llm` | ❌ **Missing** | Requires `ray[serve,llm]` extra |
| `tritonclient` | ✅ Installed | For Triton integration |
| `nvidia-pytriton` | ✅ Installed | For embedded Triton |

**Recommendation:** Use the vLLM direct approach (`deploy_vllm.py`) until you can add `ray[serve,llm]` to the dockerfile or free up disk space.
