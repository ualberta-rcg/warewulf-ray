"""
Ray Serve deployment for Stable Diffusion models
Version 2: Separate deployments per model with autoscaling
"""

import os
import asyncio
import time
import psutil
import threading
import io
import base64
from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

# PIL will be imported when needed (in the handler)

# Try to import torch separately (needed even if diffusers import fails)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

# Try to import diffusers (PIL will be imported when needed)
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  Diffusers not available. Install with: pip install diffusers>=0.21.0 accelerate>=0.20.0")


def create_deployment(model_name: str, model_path: str):
    """Create a deployment with unique name and autoscaling for a specific Stable Diffusion model"""
    
    # Create unique deployment name from model name
    deployment_name = f"sd-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
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
        # No max_queued_requests limit - allow FIFO queuing while scaling
        ray_actor_options={
            "num_gpus": 1,  # Adjust based on your GPU setup
            "runtime_env": {
                "pip": [
                    "diffusers>=0.21.0",
                    "accelerate>=0.20.0",
                    "transformers>=4.30.0",
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "pillow>=9.0.0",
                    "fastapi>=0.100.0",
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0"  # For GPU utilization metrics
                ],
                "env_vars": {
                    "HF_HOME": "/data/models",  # Use NFS for model cache
                }
            }
        }
    )
    class StableDiffusionEndpoint:
        """Stable Diffusion endpoint with autoscaling"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path):
            import sys
            import subprocess
            
            # Mark as not loaded initially (for scale-to-zero handling)
            self.model_loaded = False
            self.pipeline = None
            self.model_name = model_name
            self.model_path = model_path
            
            # Check if diffusers is available
            if not DIFFUSERS_AVAILABLE:
                print("⚠️  Diffusers not available. Attempting to install...")
                try:
                    ray_python = "/opt/ray/bin/python"
                    if os.path.exists(ray_python):
                        print("   Installing diffusers and dependencies...")
                        result = subprocess.run(
                            [ray_python, "-m", "pip", "install", 
                             "diffusers>=0.21.0", "accelerate>=0.20.0", "torchvision>=0.15.0"],
                            timeout=600  # 10 minute timeout
                        )
                        if result.returncode != 0:
                            raise RuntimeError(f"pip install failed with return code {result.returncode}")
                        
                        # Reload modules
                        import importlib
                        for mod in ['diffusers', 'torch', 'PIL']:
                            if mod in sys.modules:
                                del sys.modules[mod]
                        import torch  # Import torch first
                        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
                        print("✅ Diffusers and PyTorch installed successfully")
                    else:
                        raise ImportError("Diffusers not available and cannot install (Ray Python not found)")
                except Exception as e:
                    print(f"❌ Failed to install diffusers: {e}")
                    raise ImportError(f"Diffusers not available and installation failed: {e}")
            
            # Load model in background thread so __init__ returns quickly
            def load_model():
                try:
                    print(f"🚀 Loading Stable Diffusion model in background: {model_name}")
                    print(f"   Model path: {model_path}")
                    
                    # Determine if it's SDXL based on filename or size
                    is_sdxl = "xl" in model_path.lower() or "sd_xl" in model_path.lower()
                    
                    if is_sdxl:
                        print("   Detected SDXL model - using StableDiffusionXLPipeline")
                        self.pipeline = StableDiffusionXLPipeline.from_single_file(
                            model_path,
                            torch_dtype=torch.float16,
                            use_safetensors=True
                        )
                    else:
                        print("   Using standard StableDiffusionPipeline")
                        self.pipeline = StableDiffusionPipeline.from_single_file(
                            model_path,
                            torch_dtype=torch.float16,
                            use_safetensors=True
                        )
                    
                    # Move to GPU
                    self.pipeline = self.pipeline.to("cuda")
                    self.pipeline.enable_attention_slicing()  # Reduce memory usage
                    
                    self.model_loaded = True
                    print(f"✅ Model loaded: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load model: {e}")
                    import traceback
                    traceback.print_exc()
                    self.model_loaded = False
            
            # Start loading in background thread (non-blocking)
            loading_thread = threading.Thread(target=load_model, daemon=True)
            loading_thread.start()
            print(f"⏳ Model loading started in background for: {model_name}")
            print(f"   Handler will respond with 202 (poll) until model is ready")
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            if path.endswith("/generate") or path.endswith("/text-to-image"):
                return await self.handle_generate(request)
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
            """Health check endpoint - responds immediately even if model is loading"""
            if not hasattr(self, 'model_loaded') or not self.model_loaded:
                return JSONResponse({
                    "status": "scaling_up",
                    "model": self.model_name,
                    "message": "Model is currently loading. Please wait and retry.",
                    "ready": False
                }, status_code=503)  # Service Unavailable
            return JSONResponse({
                "status": "healthy",
                "model": self.model_name,
                "ready": True
            })
        
        async def handle_models(self, request: Request) -> JSONResponse:
            """List available models"""
            model_id = os.path.basename(self.model_path).replace(".safetensors", "").replace(".ckpt", "")
            return JSONResponse({
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "stable-diffusion"
                }]
            })
        
        async def handle_generate(self, request: Request) -> Any:
            """Handle text-to-image generation requests"""
            try:
                # Check if model is ready (scale-to-zero handling)
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'pipeline') and 
                    self.pipeline is not None
                )
                
                if not model_ready:
                    # Model is NOT ready - return poll response IMMEDIATELY
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60 seconds. Your request was not queued - please retry when ready.",
                            "model": self.model_name,
                            "estimated_ready_time_seconds": 60,
                            "retry_after": 60,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,  # Accepted - but no queuing, client must retry
                        headers={
                            "Retry-After": "60",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                
                # Start timing
                query_start_time = time.time()
                
                data = await request.json()
                prompt = data.get("prompt", "")
                negative_prompt = data.get("negative_prompt", "")
                num_inference_steps = data.get("num_inference_steps", 50)
                guidance_scale = data.get("guidance_scale", 7.5)
                width = data.get("width", 512)
                height = data.get("height", 512)
                seed = data.get("seed", None)
                return_format = data.get("format", "base64")  # base64 or url
                
                # Ensure torch is available
                if not TORCH_AVAILABLE:
                    try:
                        import torch
                    except ImportError:
                        return JSONResponse(
                            {"error": "PyTorch is not available. Please install torch."},
                            status_code=500
                        )
                
                # Generate image
                generator = None
                if seed is not None:
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                
                # Run inference
                with torch.inference_mode():
                    if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_tiling'):
                        # SDXL supports tiling for large images
                        if width > 1024 or height > 1024:
                            self.pipeline.vae.enable_tiling()
                    
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        generator=generator
                    )
                
                image = result.images[0]
                
                # Calculate query time
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                
                # Get resource usage after query
                metrics_end = self._get_resource_metrics()
                
                # Convert image to requested format
                if return_format == "base64":
                    # Convert to base64
                    from PIL import Image
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data = f"data:image/png;base64,{img_base64}"
                else:
                    # Save to file and return URL (would need file serving setup)
                    image_data = None  # Not implemented yet
                
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
                    "status": "success",
                    "model": self.model_name,
                    "image": image_data,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "parameters": {
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "width": width,
                        "height": height,
                        "seed": seed
                    },
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
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
    
    return StableDiffusionEndpoint


def create_app(model_name: str, model_path: str):
    """Create a Ray Serve application for a specific Stable Diffusion model"""
    deployment_class = create_deployment(model_name, model_path)
    return deployment_class.bind(model_name=model_name, model_path=model_path)
