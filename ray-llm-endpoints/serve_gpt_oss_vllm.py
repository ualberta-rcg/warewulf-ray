"""
Ray Serve deployment for GPT-OSS models using vLLM directly
Alternative implementation that doesn't require ray.serve.llm
"""

import sys
import os

# Ensure we use Ray from /opt/ray
if "/opt/ray/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional
import json
import asyncio

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available. Install with: pip install vllm>=0.10.1")


@serve.deployment(
    name="gpt-oss-vllm",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1,  # Adjust based on your GPU setup
        "runtime_env": {
            "pip": [
                "vllm>=0.10.1",
                "openai>=1.0.0",
            ],
            "env_vars": {
                "VLLM_USE_MODELSCOPE": "False",
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # Pass HF token via env
            }
        }
    }
)
class GPTOSSEndpoint:
    """GPT-OSS endpoint using vLLM directly"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize vLLM engine"""
        import sys
        import subprocess
        
        # Check if vLLM is available, if not try to install it
        # Initialize variables to avoid UnboundLocalError
        vllm_llm = None
        vllm_sampling_params = None
        
        if VLLM_AVAILABLE:
            # vLLM is already available
            vllm_llm = LLM
            vllm_sampling_params = SamplingParams
        else:
            print("⚠️  vLLM not available. Attempting to install...")
            try:
                # Try to install vllm using the Ray Python
                ray_python = "/opt/ray/bin/python"
                if os.path.exists(ray_python):
                    print("   Installing vllm...")
                    print("   ⏳ This can take 10-20+ minutes (downloading and compiling)")
                    print("   💡 Tip: Install manually on worker nodes to speed up:")
                    print("      /opt/ray/bin/pip install vllm>=0.10.1")
                    # Don't capture output so user can see pip progress
                    result = subprocess.run(
                        [ray_python, "-m", "pip", "install", "--upgrade", "vllm>=0.10.1"],
                        timeout=1200  # 20 minute timeout (vllm can take a while)
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"pip install failed with return code {result.returncode}")
                    
                    # Reload vllm module
                    import importlib
                    if 'vllm' in sys.modules:
                        del sys.modules['vllm']
                    from vllm import LLM, SamplingParams
                    vllm_llm = LLM
                    vllm_sampling_params = SamplingParams
                    print("✅ vLLM installed successfully")
                else:
                    raise ImportError("vLLM not available and cannot install (Ray Python not found)")
            except subprocess.TimeoutExpired:
                raise ImportError("vLLM installation timed out. Install manually with: /opt/ray/bin/pip install vllm>=0.10.1")
            except Exception as e:
                print(f"❌ Failed to install vLLM: {e}")
                raise ImportError(f"vLLM not available and installation failed: {e}. Install with: /opt/ray/bin/pip install vllm>=0.10.1")
        
        # Verify vLLM is available
        if vllm_llm is None or vllm_sampling_params is None:
            raise ImportError("vLLM is not available. Installation may have failed.")
        
        self.model_name = model_name
        print(f"🚀 Loading model: {model_name}")
        
        # Initialize vLLM engine
        # Note: This loads the model synchronously - for production, consider async loading
        try:
            # Note: max_model_len should match the model's max_position_embeddings
            # GPT-OSS-20B typically supports longer contexts, but start with 4096 to be safe
            # enforce_eager=True disables torch.compile which requires Python dev headers
            self.llm = vllm_llm(
                model=model_name,
                tensor_parallel_size=1,  # Adjust for multi-GPU
                max_model_len=4096,  # Start with 4096, can increase if model supports it
                enforce_eager=True,  # Disable torch.compile (avoids Python.h requirement)
            )
            print(f"✅ Model loaded: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def __call__(self, request: Request) -> Any:
        """Handle OpenAI-compatible API requests"""
        path = request.url.path
        
        # Route to appropriate handler
        if path.endswith("/chat/completions"):
            return await self.handle_chat_completions(request)
        elif path.endswith("/models"):
            return await self.handle_models(request)
        elif path.endswith("/health"):
            return await self.handle_health(request)
        else:
            return JSONResponse(
                {"error": f"Unknown endpoint: {path}"},
                status_code=404
            )
    
    async def handle_health(self, request: Request) -> JSONResponse:
        """Health check endpoint"""
        return JSONResponse({"status": "healthy", "model": self.model_name})
    
    async def handle_models(self, request: Request) -> JSONResponse:
        """List available models"""
        model_id = self.model_name.replace("/", "-")
        return JSONResponse({
            "data": [{
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "huggingface"
            }]
        })
    
    async def handle_chat_completions(self, request: Request) -> Any:
        """Handle chat completion requests"""
        try:
            data = await request.json()
            messages = data.get("messages", [])
            stream = data.get("stream", False)
            max_tokens = data.get("max_tokens", 100)
            temperature = data.get("temperature", 0.7)
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if stream:
                # Streaming response
                return StreamingResponse(
                    self._generate_stream(prompt, sampling_params),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                outputs = self.llm.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                return JSONResponse({
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self.model_name.replace("/", "-"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(generated_text.split()),
                        "total_tokens": len(prompt.split()) + len(generated_text.split())
                    }
                })
                
        except Exception as e:
            return JSONResponse(
                {"error": str(e), "type": type(e).__name__},
                status_code=500
            )
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert OpenAI messages format to prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def _generate_stream(self, prompt: str, sampling_params):
        """Generate streaming response"""
        # For streaming, we need to use async engine
        # This is a simplified version - for production, use AsyncLLMEngine
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Simulate streaming by chunking the response
        words = generated_text.split()
        for i, word in enumerate(words):
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model_name.replace("/", "-"),
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        yield "data: [DONE]\n\n"


# Create deployment
def create_app(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Create the GPT-OSS application"""
    return GPTOSSEndpoint.bind(model_name=model_name)

# Default app
app = create_app()

if __name__ == "__main__":
    print("✅ GPT-OSS vLLM endpoint configured")
    print("   This version uses vLLM directly (no ray.serve.llm needed)")
    print("   Deploy with: python deploy_vllm.py")
