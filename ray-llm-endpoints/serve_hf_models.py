"""
Ray Serve deployment for HuggingFace models using vLLM directly
Works without ray.serve.llm - uses vLLM's OpenAI-compatible server
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
import subprocess
import signal

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available. Install with: pip install vllm>=0.10.1")


@serve.deployment(
    name="hf-llm-vllm",
    num_replicas=1,
    ray_actor_options={
        "num_gpus": 1,  # Adjust based on your GPU setup
        "runtime_env": {
            "pip": [
                "vllm>=0.10.1",
                "openai>=1.0.0",
                "transformers>=4.30.0",
                "torch>=2.0.0"
            ],
            "env_vars": {
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),  # Pass HF token via env
            }
        }
    }
)
class HuggingFaceLLMEndpoint:
    """HuggingFace LLM endpoint using vLLM directly"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat", hf_token: str = None):
        """Initialize vLLM engine with HuggingFace model"""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available. Install with: pip install vllm>=0.10.1")
        
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        
        print(f"🚀 Loading HuggingFace model: {model_name}")
        
        # Set HuggingFace token if provided
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
        
        # Initialize vLLM engine
        # Note: This loads the model synchronously - for production, consider async loading
        try:
            # Configure vLLM with HuggingFace token if provided
            llm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": 1,  # Adjust for multi-GPU
                "max_model_len": 32768,  # Adjust based on GPU memory
                "trust_remote_code": True,  # For models that need custom code
                "download_dir": "/data/models",  # Cache models on NFS
            }
            
            # Set HuggingFace token for authentication
            if self.hf_token:
                llm_kwargs["token"] = self.hf_token
            
            self.llm = LLM(**llm_kwargs)
            print(f"✅ Model loaded: {model_name}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print(f"   Make sure you have:")
            print(f"   1. Sufficient disk space (check: df -h)")
            print(f"   2. Network access to huggingface.co")
            if self.hf_token:
                print(f"   3. Valid HuggingFace token")
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
        return JSONResponse({
            "status": "healthy",
            "model": self.model_name,
            "vllm_available": VLLM_AVAILABLE
        })
    
    async def handle_models(self, request: Request) -> JSONResponse:
        """List available models"""
        return JSONResponse({
            "data": [{
                "id": self.model_name.replace("/", "-"),
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
            
            # Convert messages to prompt (vLLM format)
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
        # For chat models, use proper chat template
        # This is a simplified version - vLLM handles chat templates automatically
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
    
    async def _generate_stream(self, prompt: str, sampling_params: SamplingParams):
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
def create_app(model_name: str = "deepseek-ai/deepseek-llm-7b-chat", hf_token: str = None):
    """Create the HuggingFace LLM application"""
    return HuggingFaceLLMEndpoint.bind(model_name=model_name, hf_token=hf_token)

# Default app
app = create_app()

if __name__ == "__main__":
    print("✅ HuggingFace LLM vLLM endpoint configured")
    print("   This version uses vLLM directly (no ray.serve.llm needed)")
    print("   Deploy with: python deploy_hf_models.py")
