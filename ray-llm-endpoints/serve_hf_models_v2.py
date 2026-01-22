"""
Ray Serve deployment for HuggingFace LLM models using vLLM directly
Version 2: Separate deployments per model with autoscaling and dynamic input discovery
"""

import os
import sys
import asyncio
import time
import psutil
import threading
import inspect
from typing import Any, Dict
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

# Note: We don't import vllm at module level to avoid library registration conflicts
# All imports happen inside the replica's __init__ method where they're needed
# This prevents issues when Ray creates multiple replicas in the same process


def create_deployment(model_name: str, model_path: str, hf_token: str = None, max_model_len: int = 4096):
    """Create a deployment with unique name and autoscaling for a specific model"""
    
    # Create unique deployment name from model name
    deployment_name = f"hf-llm-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    @serve.deployment(
        name=deployment_name,
        autoscaling_config={
            "min_replicas": 0,  # Scale to zero
            "max_replicas": 2,  # Max 2 replicas per model
            "target_num_ongoing_requests_per_replica": 1,
            "initial_replicas": 0,
            "downscale_delay_s": 300,  # Wait 5 minutes before scaling down
            "upscale_delay_s": 0,  # Scale up immediately
            "max_queued_requests": 0,  # Don't queue - return immediately when scaling up
        },
        ray_actor_options={
            "num_gpus": 1,
            "runtime_env": {
                "pip": [
                    # CRITICAL: packaging MUST be first - Ray's verification needs it immediately
                    "packaging>=21.0",  # Required by Ray for runtime env verification - install FIRST
                    # Core build dependencies (needed for package installation)
                    "setuptools>=65.0",  # Required for building packages
                    "wheel>=0.38.0",  # Required for building packages
                    # Install Ray in venv to match system version (needed for verification)
                    "ray[serve]>=2.49.0",  # Match system Ray version for venv compatibility
                    # ML/AI framework dependencies
                    "numpy>=1.24.0",  # Required by torch and vllm
                    # Install opencv-python-headless explicitly from PyPI before vllm
                    # This avoids ComputeCanada dummy package conflict
                    "opencv-python-headless>=4.11.0",
                    "vllm>=0.10.1",
                    "torch>=2.0.0",
                    "transformers>=4.30.0",
                    "accelerate>=0.20.0",
                    # OpenAI API compatibility
                    "openai>=1.0.0",
                    # Web framework
                    "fastapi>=0.100.0",
                    "starlette>=0.27.0",  # FastAPI dependency
                    "uvicorn>=0.23.0",  # ASGI server
                    # System utilities
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0",  # GPU monitoring
                ],
                "env_vars": {
                    "HF_HOME": "/data/models",
                    "VLLM_USE_MODELSCOPE": "False",
                    # Force pip to use PyPI to avoid ComputeCanada dummy packages
                    "PIP_INDEX_URL": "https://pypi.org/simple",
                    "PIP_EXTRA_INDEX_URL": "",  # Clear any ComputeCanada indexes
                    # Token will be passed as parameter and set in worker's __init__
                }
            }
        }
    )
    class HuggingFaceLLMEndpoint:
        """HuggingFace LLM endpoint using vLLM directly with autoscaling and dynamic input discovery"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path, hf_token: str = None, max_model_len: int = max_model_len):
            self.model_loaded = False
            self.llm = None  # vLLM LLM instance
            self.model_name = model_name
            self.model_path = model_path or model_name
            self.max_model_len = max_model_len
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            
            # Get HF token from parameter, or fall back to environment
            self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Set token in environment for vLLM library
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
                print(f"✅ HuggingFace token configured (length: {len(self.hf_token)})")
            else:
                print("⚠️  No HuggingFace token found - may fail for gated models")
            
            # Import vllm (it's in runtime_env, so should be available)
            try:
                from vllm import LLM, SamplingParams
                self.vllm_llm_class = LLM
                self.vllm_sampling_params_class = SamplingParams
                print("✅ vLLM available")
            except ImportError as import_err:
                print(f"⚠️  vLLM not available: {import_err}")
                print("   Attempting to install...")
                try:
                    # Use the replica's Python (from runtime_env venv)
                    replica_python = sys.executable
                    print(f"   Using Python: {replica_python}")
                    import subprocess
                    # Install vLLM with explicit PyPI index to avoid ComputeCanada package conflicts
                    result = subprocess.run(
                        [replica_python, "-m", "pip", "install", 
                         "--index-url", "https://pypi.org/simple",
                         "vllm>=0.10.1"],
                        timeout=1200,  # 20 minute timeout
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        print(f"   pip install stderr: {result.stderr[:500]}")
                        raise RuntimeError(f"pip install failed with return code {result.returncode}")
                    
                    # Import after installation
                    from vllm import LLM, SamplingParams
                    self.vllm_llm_class = LLM
                    self.vllm_sampling_params_class = SamplingParams
                    print("✅ Installed and imported vLLM")
                except Exception as install_err:
                    print(f"❌ Installation failed: {install_err}")
                    raise ImportError(f"vLLM not available and installation failed: {install_err}")
            
            # Load model in background thread
            def load_model():
                try:
                    from vllm import LLM, SamplingParams
                    
                    self.loading_started = True
                    print(f"🚀 Loading model in background: {model_name}")
                    print(f"   HuggingFace Model ID: {self.model_path}")
                    print(f"   Max context length: {max_model_len} tokens")
                    
                    # Configure vLLM
                    llm_kwargs = {
                        "model": self.model_path,
                        "tensor_parallel_size": 1,  # Adjust for multi-GPU
                        "max_model_len": max_model_len,
                        "trust_remote_code": True,
                        "download_dir": "/data/models",  # Cache models on NFS
                        "enforce_eager": True,  # Disable torch.compile (avoids Python.h requirement)
                    }
                    
                    # Set HuggingFace token for authentication
                    if self.hf_token:
                        llm_kwargs["token"] = self.hf_token
                    
                    # Load model
                    print(f"   Loading model from HuggingFace: {self.model_path}")
                    self.llm = self.vllm_llm_class(**llm_kwargs)
                    
                    # Verify GPU usage
                    try:
                        import torch
                        if torch.cuda.is_available():
                            num_cuda_devices = torch.cuda.device_count()
                            print(f"   ✅ CUDA available: {num_cuda_devices} device(s)")
                            for i in range(num_cuda_devices):
                                allocated = torch.cuda.memory_allocated(i) / 1024**3
                                reserved = torch.cuda.memory_reserved(i) / 1024**3
                                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                                print(f"   GPU {i} ({torch.cuda.get_device_name(i)}):")
                                print(f"      Memory allocated: {allocated:.2f} GB / {total:.2f} GB")
                                print(f"      Memory reserved: {reserved:.2f} GB")
                                print(f"      Memory free: {total - reserved:.2f} GB")
                        else:
                            print("   ⚠️  CUDA not available - model may be on CPU")
                    except Exception as gpu_check_err:
                        print(f"   ⚠️  Could not verify GPU: {gpu_check_err}")
                    
                    # Discover schema after model is loaded
                    self.schema = self._discover_model_schema()
                    print(f"✅ Model loaded: {model_name}")
                    print(f"   Discovered {len(self.schema.get('inputs', {}).get('required', []))} required inputs")
                    print(f"   Discovered {len(self.schema.get('inputs', {}).get('optional', []))} optional inputs")
                    
                    self.model_loaded = True
                    self.loading_error = None  # Clear any previous errors
                    print(f"✅ Model successfully loaded: {model_name}")
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Failed to load model: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    self.model_loaded = False
                    self.loading_error = error_msg  # Store error for API responses
            
            loading_thread = threading.Thread(target=load_model, daemon=True)
            loading_thread.start()
            print(f"⏳ Model loading started in background for: {model_name}")
        
        def _discover_model_schema(self) -> Dict[str, Any]:
            """Dynamically discover the model's inputs and capabilities"""
            if self.llm is None:
                return {"error": "Model not loaded"}
            
            schema = {
                "model_id": self.model_name,
                "model_class": "vLLM",
                "capabilities": ["text-generation", "chat-completion"],
                "inputs": {
                    "required": [],
                    "optional": []
                },
                "defaults": {
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "stream": False,
                }
            }
            
            # OpenAI-compatible chat completion API
            schema["inputs"]["required"] = ["messages"]
            
            # Optional: generation parameters
            schema["inputs"]["optional"] = [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stream",
                "stop",
                "presence_penalty",
                "frequency_penalty",
            ]
            
            return schema
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            # For generation endpoints, check readiness immediately before any processing
            if path.endswith("/chat/completions"):
                # Quick readiness check - return immediately if not ready
                if not (hasattr(self, 'model_loaded') and self.model_loaded and 
                        hasattr(self, 'llm') and self.llm is not None):
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60-120 seconds.",
                            "model": getattr(self, 'model_name', 'unknown'),
                            "estimated_ready_time_seconds": 120,
                            "retry_after": 120,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "120",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                return await self.handle_chat_completions(request)
            elif path.endswith("/schema"):
                return await self.handle_schema(request)
            elif path.endswith("/models"):
                return await self.handle_models(request)
            elif path.endswith("/health"):
                return await self.handle_health(request)
            else:
                return JSONResponse(
                    {"error": f"Unknown endpoint: {path}"},
                    status_code=404
                )
        
        async def handle_schema(self, request: Request) -> JSONResponse:
            """Return the model's input schema (discovered dynamically)"""
            if not hasattr(self, 'llm') or self.llm is None:
                response = {
                    "error": "Model not loaded yet",
                    "status": "scaling_up"
                }
                
                # Include loading status
                if hasattr(self, 'loading_started'):
                    if self.loading_started:
                        response["status"] = "loading"
                    else:
                        response["status"] = "initializing"
                
                # Include error if loading failed
                if hasattr(self, 'loading_error') and self.loading_error:
                    response["loading_error"] = self.loading_error
                    response["error"] = f"Model loading failed: {self.loading_error}"
                    response["status"] = "failed"
                
                return JSONResponse(response, status_code=503)
            
            if self.schema is None:
                # Try to discover now
                self.schema = self._discover_model_schema()
            
            return JSONResponse(self.schema)
        
        async def handle_health(self, request: Request) -> JSONResponse:
            """Health check endpoint"""
            if not hasattr(self, 'model_loaded') or not self.model_loaded:
                return JSONResponse({
                    "status": "scaling_up",
                    "model": self.model_name,
                    "message": "Model is currently loading. Please wait and retry.",
                    "ready": False
                }, status_code=503)
            return JSONResponse({
                "status": "healthy",
                "model": self.model_name,
                "ready": True
            })
        
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
            """Handle chat completion requests (OpenAI-compatible)"""
            try:
                # Check if model is ready
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'llm') and 
                    self.llm is not None
                )
                
                if not model_ready:
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60-120 seconds.",
                            "model": self.model_name,
                            "estimated_ready_time_seconds": 120,
                            "retry_after": 120,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "120",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                
                query_start_time = time.time()
                
                data = await request.json()
                messages = data.get("messages", [])
                stream = data.get("stream", False)
                max_tokens = data.get("max_tokens", 100)
                temperature = data.get("temperature", 0.7)
                
                # Convert messages to prompt
                prompt = self._messages_to_prompt(messages)
                
                # Create sampling params
                sampling_params = self.vllm_sampling_params_class(
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                if stream:
                    # Streaming response (metrics added at end)
                    return StreamingResponse(
                        self._generate_stream(prompt, sampling_params, query_start_time),
                        media_type="text/event-stream"
                    )
                else:
                    # Non-streaming response
                    outputs = self.llm.generate([prompt], sampling_params)
                    generated_text = outputs[0].outputs[0].text
                    
                    # Calculate query time
                    query_end_time = time.time()
                    query_duration = query_end_time - query_start_time
                    
                    # Get resource usage after query (from replica)
                    metrics_end = self._get_resource_metrics()
                    
                    # Calculate resource usage (replica-specific metrics)
                    compute_metrics = {
                        "query_duration_seconds": round(query_duration, 3),
                        "replica_cpu_percent": metrics_end.get("cpu_percent", 0),
                        "replica_memory_mb": metrics_end.get("memory_mb", 0),
                        "replica_num_cpus": metrics_end.get("num_cpus", 1),
                        "replica_gpu_memory_allocated_mb": metrics_end.get("gpu_memory_allocated_mb", 0),
                        "replica_gpu_memory_reserved_mb": metrics_end.get("gpu_memory_reserved_mb", 0),
                        "replica_gpu_memory_total_mb": metrics_end.get("gpu_memory_total_mb", 0),
                        "replica_gpu_memory_used_mb": metrics_end.get("gpu_memory_used_mb", metrics_end.get("gpu_memory_allocated_mb", 0)),
                        "replica_gpu_utilization_percent": metrics_end.get("gpu_utilization_percent", 0),
                    }
                    
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
                        },
                        "compute_metrics": compute_metrics
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
        
        def _get_resource_metrics(self):
            """Get current resource usage metrics from the replica"""
            metrics = {}
            try:
                # Get current process
                process = psutil.Process()
                
                # CPU usage - get per-process CPU percent
                metrics["cpu_percent"] = round(process.cpu_percent(interval=0.1), 2)
                
                # Memory usage (RSS - Resident Set Size)
                memory_info = process.memory_info()
                metrics["memory_mb"] = round(memory_info.rss / (1024 * 1024), 2)
                
                # Number of CPUs used by this process
                metrics["num_cpus"] = len(process.cpu_affinity()) if hasattr(process, 'cpu_affinity') else 1
                
                # GPU metrics (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.cuda.current_device()
                        
                        # GPU memory allocated (actually used by tensors)
                        gpu_allocated = torch.cuda.memory_allocated(device)
                        metrics["gpu_memory_allocated_mb"] = round(gpu_allocated / (1024 * 1024), 2)
                        
                        # GPU memory reserved (allocated by PyTorch)
                        gpu_reserved = torch.cuda.memory_reserved(device)
                        metrics["gpu_memory_reserved_mb"] = round(gpu_reserved / (1024 * 1024), 2)
                        
                        # Total GPU memory
                        gpu_total = torch.cuda.get_device_properties(device).total_memory
                        metrics["gpu_memory_total_mb"] = round(gpu_total / (1024 * 1024), 2)
                        
                        # GPU utilization (requires nvidia-ml-py)
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            metrics["gpu_utilization_percent"] = util.gpu
                            
                            # GPU memory info from NVML (more accurate)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            metrics["gpu_memory_used_mb"] = round(mem_info.used / (1024 * 1024), 2)
                            metrics["gpu_memory_free_mb"] = round(mem_info.free / (1024 * 1024), 2)
                        except:
                            # Fallback if pynvml not available
                            metrics["gpu_utilization_percent"] = None
                            metrics["gpu_memory_used_mb"] = None
                            metrics["gpu_memory_free_mb"] = None
                    else:
                        metrics["gpu_memory_allocated_mb"] = 0
                        metrics["gpu_memory_reserved_mb"] = 0
                        metrics["gpu_memory_total_mb"] = 0
                        metrics["gpu_utilization_percent"] = 0
                except ImportError:
                    metrics["gpu_memory_allocated_mb"] = 0
                    metrics["gpu_memory_reserved_mb"] = 0
                    metrics["gpu_memory_total_mb"] = 0
                    metrics["gpu_utilization_percent"] = 0
            except Exception as e:
                # If metrics collection fails, return zeros
                metrics = {
                    "cpu_percent": 0,
                    "memory_mb": 0,
                    "num_cpus": 0,
                    "gpu_memory_allocated_mb": 0,
                    "gpu_memory_reserved_mb": 0,
                    "gpu_utilization_percent": 0
                }
            
            return metrics
        
        async def _generate_stream(self, prompt: str, sampling_params, query_start_time):
            """Generate streaming response with metrics"""
            # For streaming, we need to use async engine
            # This is a simplified version - for production, use AsyncLLMEngine
            outputs = self.llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Calculate query time
            query_end_time = time.time()
            query_duration = query_end_time - query_start_time
            
            # Get resource usage
            metrics = self._get_resource_metrics()
            compute_metrics = {
                "query_duration_seconds": round(query_duration, 3),
                "replica_cpu_percent": metrics.get("cpu_percent", 0),
                "replica_memory_mb": metrics.get("memory_mb", 0),
                "replica_num_cpus": metrics.get("num_cpus", 1),
                "replica_gpu_memory_allocated_mb": metrics.get("gpu_memory_allocated_mb", 0),
                "replica_gpu_memory_reserved_mb": metrics.get("gpu_memory_reserved_mb", 0),
                "replica_gpu_memory_total_mb": metrics.get("gpu_memory_total_mb", 0),
                "replica_gpu_memory_used_mb": metrics.get("gpu_memory_used_mb", metrics.get("gpu_memory_allocated_mb", 0)),
                "replica_gpu_utilization_percent": metrics.get("gpu_utilization_percent", 0),
            }
            
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
            
            # Send metrics as final chunk
            metrics_chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model_name.replace("/", "-"),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "compute_metrics": compute_metrics
            }
            yield f"data: {metrics_chunk}\n\n"
            yield "data: [DONE]\n\n"
    
    return HuggingFaceLLMEndpoint


def create_app(model_name: str, model_path: str = None, hf_token: str = None, max_model_len: int = 4096):
    """Create a Ray Serve application for a specific model"""
    if model_path is None:
        model_path = model_name
    deployment_class = create_deployment(model_name, model_path, hf_token=hf_token, max_model_len=max_model_len)
    return deployment_class.bind(model_name=model_name, model_path=model_path, hf_token=hf_token, max_model_len=max_model_len)
