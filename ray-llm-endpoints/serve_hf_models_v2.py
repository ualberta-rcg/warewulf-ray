"""
Ray Serve deployment for HuggingFace LLM models using vLLM directly
Version 2: Separate deployments per model with autoscaling
"""

import os
import asyncio
from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

# Try to import vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not available. Install with: pip install vllm>=0.10.1")


def create_deployment(model_name: str, max_model_len: int = 4096):
    """Create a deployment with unique name and autoscaling for a specific model"""
    
    # Create unique deployment name from model name
    deployment_name = f"hf-llm-{model_name.replace('/', '-').replace('_', '-')}"
    
    @serve.deployment(
        name=deployment_name,
        autoscaling_config={
            "min_replicas": 0,  # Scale to zero
            "max_replicas": 1,  # Max 1 replica per model
            "target_num_ongoing_requests_per_replica": 1,  # Scale up when 1 request is queued
            "initial_replicas": 0,  # Start with 0 replicas
            "downscale_delay_s": 300,  # Wait 5 minutes before scaling down to zero
            "upscale_delay_s": 0,  # Scale up immediately when request arrives
        },
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
                    "VLLM_USE_MODELSCOPE": "False",
                }
            }
        }
    )
    class HuggingFaceLLMEndpoint:
        """HuggingFace LLM endpoint using vLLM directly with autoscaling"""
        
        def __init__(self, model_name: str = model_name, hf_token: str = None, max_model_len: int = max_model_len):
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
            self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"🚀 Loading model: {model_name}")
            print(f"   Max context length: {max_model_len}")
            
            # Set HuggingFace token if provided
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
            
            # Initialize vLLM engine
            # Note: This loads the model synchronously - for production, consider async loading
            try:
                # Configure vLLM with HuggingFace token if provided
                # enforce_eager=True disables torch.compile which requires Python dev headers
                llm_kwargs = {
                    "model": model_name,
                    "tensor_parallel_size": 1,  # Adjust for multi-GPU
                    "max_model_len": max_model_len,  # Configurable per model
                    "trust_remote_code": True,  # For models that need custom code
                    "download_dir": "/data/models",  # Cache models on NFS
                    "enforce_eager": True,  # Disable torch.compile (avoids Python.h requirement)
                }
                
                # Set HuggingFace token for authentication
                if self.hf_token:
                    llm_kwargs["token"] = self.hf_token
                
                # Use the locally imported LLM class
                self.llm = vllm_llm(**llm_kwargs)
                print(f"✅ Model loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Make sure you have:")
                print(f"   1. Sufficient disk space (check: df -h)")
                print(f"   2. Network access to huggingface.co")
                if self.hf_token:
                    print(f"   3. Valid HuggingFace token")
                raise
        
        async def __call__(self, request: Request) -> Any:
            """Handle OpenAI-compatible API requests"""
            path = request.url.path
            
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
                    {"error": str(e)},
                    status_code=500
                )
        
        def _messages_to_prompt(self, messages):
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
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            yield "data: [DONE]\n\n"
    
    return HuggingFaceLLMEndpoint


def create_app(model_name: str, hf_token: str = None, max_model_len: int = 4096):
    """Create a Ray Serve application for a specific model"""
    deployment_class = create_deployment(model_name, max_model_len)
    return deployment_class.bind(model_name=model_name, hf_token=hf_token, max_model_len=max_model_len)
