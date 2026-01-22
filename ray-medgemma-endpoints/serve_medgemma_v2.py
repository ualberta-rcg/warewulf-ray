"""
Ray Serve deployment for MedGemma models
Version 2: Separate deployments per model with autoscaling and dynamic input discovery
"""

import os
import sys
import asyncio
import time
import psutil
import threading
import io
import base64
import inspect
from typing import Any, Dict, List, Optional
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

# Note: We don't import torch/transformers at module level to avoid Triton library registration conflicts
# All imports happen inside the replica's __init__ method where they're needed
# This prevents issues when Ray creates multiple replicas in the same process


def create_deployment(model_name: str, model_path: str, hf_token: str = None, num_gpus: int = 1):
    """Create a deployment with unique name and autoscaling for a specific MedGemma model"""
    
    # Create unique deployment name from model name
    deployment_name = f"medgemma-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    @serve.deployment(
        name=deployment_name,
        autoscaling_config={
            "min_replicas": 0,  # Scale to zero
            "max_replicas": 2,  # Max 2 replicas per model (27B is large)
            "target_num_ongoing_requests_per_replica": 1,
            "initial_replicas": 0,
            "downscale_delay_s": 300,  # Wait 5 minutes before scaling down
            "upscale_delay_s": 0,  # Scale up immediately
            "max_queued_requests": 0,  # Don't queue - return immediately when scaling up
        },
        ray_actor_options={
            "num_gpus": num_gpus,  # Configurable number of GPUs (27B model may need 2-4 GPUs)
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
                    "torch>=2.0.0",
                    "transformers>=4.50.0",  # Required for Gemma 3 / MedGemma support
                    "accelerate>=0.20.0",
                    # Image processing
                    "pillow>=9.0.0",
                    # Web framework
                    "fastapi>=0.100.0",
                    "starlette>=0.27.0",  # FastAPI dependency
                    "uvicorn>=0.23.0",  # ASGI server
                    # System utilities
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0",  # GPU monitoring
                    # Optional: for quantization (if needed for 27B model)
                    "bitsandbytes>=0.41.0",  # For 4-bit/8-bit quantization
                ],
                "env_vars": {
                    "HF_HOME": "/data/models",
                    # Token will be passed as parameter and set in worker's __init__
                }
            }
        }
    )
    class MedGemmaEndpoint:
        """MedGemma endpoint with autoscaling and dynamic input discovery"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path, hf_token: str = None, num_gpus_param: int = None):
            # Use provided num_gpus_param or fall back to the one from closure
            self.num_gpus = num_gpus_param if num_gpus_param is not None else num_gpus
            self.model_loaded = False
            self.model = None  # AutoModelForImageTextToText
            self.processor = None  # AutoProcessor
            self.model_name = model_name
            self.model_path = model_path
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            
            # Get HF token from parameter, or fall back to environment
            self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            
            # Set token in environment for transformers library
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
                print(f"✅ HuggingFace token configured (length: {len(self.hf_token)})")
            else:
                print("⚠️  No HuggingFace token found - may fail for gated models")
            
            # Import torch and transformers (they're in runtime_env, so should be available)
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
                print("✅ PyTorch and Transformers available")
            except ImportError as import_err:
                print(f"⚠️  PyTorch/Transformers not available: {import_err}")
                print("   Attempting to install...")
                try:
                    # Use the replica's Python (from runtime_env venv)
                    replica_python = sys.executable
                    print(f"   Using Python: {replica_python}")
                    import subprocess
                    result = subprocess.run(
                        [replica_python, "-m", "pip", "install", 
                         "transformers>=4.50.0", "accelerate>=0.20.0", "torch>=2.0.0"],
                        timeout=600,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"pip install failed with return code {result.returncode}")
                    
                    # Import after installation
                    import torch
                    from transformers import AutoModelForImageTextToText, AutoProcessor
                    print("✅ Installed and imported PyTorch/Transformers")
                except Exception as install_err:
                    print(f"❌ Installation failed: {install_err}")
                    raise ImportError(f"PyTorch/Transformers not available and installation failed: {install_err}")
            
            # Load model in background thread
            def load_model():
                try:
                    # Import torch in the function scope to avoid UnboundLocalError
                    import torch
                    from transformers import AutoModelForImageTextToText, AutoProcessor
                    
                    self.loading_started = True
                    print(f"🚀 Loading MedGemma model in background: {model_name}")
                    print(f"   HuggingFace Model ID: {model_path}")
                    
                    # Load processor first (lightweight)
                    print(f"   Loading processor from HuggingFace: {model_path}")
                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        token=self.hf_token
                    )
                    print("   ✅ Processor loaded")
                    
                    # Load model with appropriate settings for 27B model
                    print(f"   Loading model from HuggingFace: {model_path}")
                    print(f"   ⚠️  Note: 27B model is large - using {self.num_gpus} GPU(s)")
                    if self.num_gpus > 1:
                        print(f"   ✅ Multi-GPU mode: model will be split across {self.num_gpus} GPUs")
                    else:
                        print("   ⚠️  Single GPU mode: model may use CPU offloading if GPU memory is insufficient")
                    
                    # Try to load with bfloat16 (recommended for Gemma models)
                    try:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",  # Automatically handle device placement
                            trust_remote_code=True,
                            token=self.hf_token
                        )
                        print("   ✅ Model loaded with bfloat16 and device_map='auto'")
                    except Exception as bf16_err:
                        print(f"   ⚠️  Failed to load with bfloat16: {bf16_err}")
                        print("   Trying with float16...")
                        try:
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                trust_remote_code=True,
                                token=self.hf_token
                            )
                            print("   ✅ Model loaded with float16 and device_map='auto'")
                        except Exception as f16_err:
                            print(f"   ⚠️  Failed to load with float16: {f16_err}")
                            print("   Trying with default dtype (may use more memory)...")
                            # Final fallback: let transformers decide
                            self.model = AutoModelForImageTextToText.from_pretrained(
                                model_path,
                                device_map="auto",
                                trust_remote_code=True,
                                token=self.hf_token
                            )
                            print("   ✅ Model loaded with default dtype")
                    
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
            if self.model is None or self.processor is None:
                return {"error": "Model not loaded"}
            
            schema = {
                "model_id": self.model_name,
                "model_class": type(self.model).__name__,
                "processor_class": type(self.processor).__name__,
                "capabilities": ["text-to-text", "image-text-to-text"],
                "inputs": {
                    "required": [],
                    "optional": []
                },
                "defaults": {
                    "max_new_tokens": 200,
                    "do_sample": False,
                }
            }
            
            # MedGemma supports both text-only and image+text inputs
            # Required: messages (chat format)
            schema["inputs"]["required"] = ["messages"]
            
            # Optional: generation parameters
            schema["inputs"]["optional"] = [
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
            ]
            
            return schema
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            # For generation endpoints, check readiness immediately before any processing
            if path.endswith("/generate") or path.endswith("/chat"):
                # Quick readiness check - return immediately if not ready
                if not (hasattr(self, 'model_loaded') and self.model_loaded and 
                        hasattr(self, 'model') and self.model is not None):
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60-120 seconds (27B model takes longer to load).",
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
                return await self.handle_generate(request)
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
            if not hasattr(self, 'model') or self.model is None:
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
            return JSONResponse({
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "medgemma"
                }]
            })
        
        async def handle_generate(self, request: Request) -> Any:
            """Handle text generation requests (text-only or image+text)"""
            try:
                # Check if model is ready
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'model') and 
                    self.model is not None and
                    hasattr(self, 'processor') and 
                    self.processor is not None
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
                
                # Get request data
                data = await request.json()
                
                # Ensure torch is available
                try:
                    import torch
                except ImportError:
                    return JSONResponse(
                        {"error": "PyTorch is not available. Please install torch."},
                        status_code=500
                    )
                
                # Build messages from request
                messages = self._build_messages(data)
                
                # Get generation parameters
                max_new_tokens = data.get("max_new_tokens", 200)
                do_sample = data.get("do_sample", False)
                temperature = data.get("temperature", None)
                top_p = data.get("top_p", None)
                top_k = data.get("top_k", None)
                repetition_penalty = data.get("repetition_penalty", None)
                
                # Prepare inputs using processor
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move inputs to model device
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device, dtype=torch.bfloat16 if v.dtype == torch.float32 else v.dtype) 
                         if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                input_len = inputs["input_ids"].shape[-1]
                
                # Build generation kwargs
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if temperature is not None:
                    generation_kwargs["temperature"] = temperature
                if top_p is not None:
                    generation_kwargs["top_p"] = top_p
                if top_k is not None:
                    generation_kwargs["top_k"] = top_k
                if repetition_penalty is not None:
                    generation_kwargs["repetition_penalty"] = repetition_penalty
                
                # Generate text
                with torch.inference_mode():
                    generation = self.model.generate(**inputs, **generation_kwargs)
                    generation = generation[0][input_len:]
                
                # Decode generated text
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                metrics_end = self._get_resource_metrics()
                
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
                    "status": "success",
                    "model": self.model_name,
                    "response": decoded,
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                    },
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
        def _build_messages(self, data: Dict) -> List[Dict]:
            """Build chat messages from request data"""
            messages = []
            
            # If messages are provided directly, use them
            if "messages" in data:
                messages = data["messages"]
            else:
                # Build messages from simpler format
                system_prompt = data.get("system", "You are a helpful medical assistant.")
                user_content = []
                
                # Add text if provided
                if "text" in data or "prompt" in data:
                    text = data.get("text") or data.get("prompt")
                    user_content.append({"type": "text", "text": text})
                
                # Add image if provided
                if "image" in data:
                    image = self._decode_base64_image(data["image"])
                    user_content.append({"type": "image", "image": image})
                
                # Build message structure
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    })
                
                if user_content:
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
            
            return messages
        
        def _decode_base64_image(self, image_data: str):
            """Decode base64 image data"""
            from PIL import Image
            
            # Handle data URI format: "data:image/png;base64,..."
            if image_data.startswith("data:"):
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        
        def _get_resource_metrics(self):
            """Get current resource usage metrics from the replica"""
            metrics = {}
            try:
                process = psutil.Process()
                metrics["cpu_percent"] = round(process.cpu_percent(interval=0.1), 2)
                memory_info = process.memory_info()
                metrics["memory_mb"] = round(memory_info.rss / (1024 * 1024), 2)
                metrics["num_cpus"] = len(process.cpu_affinity()) if hasattr(process, 'cpu_affinity') else 1
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = torch.cuda.current_device()
                        gpu_allocated = torch.cuda.memory_allocated(device)
                        metrics["gpu_memory_allocated_mb"] = round(gpu_allocated / (1024 * 1024), 2)
                        gpu_reserved = torch.cuda.memory_reserved(device)
                        metrics["gpu_memory_reserved_mb"] = round(gpu_reserved / (1024 * 1024), 2)
                        gpu_total = torch.cuda.get_device_properties(device).total_memory
                        metrics["gpu_memory_total_mb"] = round(gpu_total / (1024 * 1024), 2)
                        
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            metrics["gpu_utilization_percent"] = util.gpu
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            metrics["gpu_memory_used_mb"] = round(mem_info.used / (1024 * 1024), 2)
                            metrics["gpu_memory_free_mb"] = round(mem_info.free / (1024 * 1024), 2)
                        except:
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
                metrics = {
                    "cpu_percent": 0,
                    "memory_mb": 0,
                    "num_cpus": 0,
                    "gpu_memory_allocated_mb": 0,
                    "gpu_memory_reserved_mb": 0,
                    "gpu_utilization_percent": 0
                }
            
            return metrics
    
    return MedGemmaEndpoint


def create_app(model_name: str, model_path: str, hf_token: str = None, num_gpus: int = 1):
    """Create a Ray Serve application for a specific MedGemma model"""
    deployment_class = create_deployment(model_name, model_path, hf_token=hf_token, num_gpus=num_gpus)
    return deployment_class.bind(model_name=model_name, model_path=model_path, hf_token=hf_token, num_gpus_param=num_gpus)
