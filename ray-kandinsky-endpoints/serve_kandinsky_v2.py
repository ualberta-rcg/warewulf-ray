"""
Ray Serve deployment for Kandinsky models
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

# Note: We don't import torch/diffusers at module level to avoid Triton library registration conflicts
# All imports happen inside the replica's __init__ method where they're needed
# This prevents issues when Ray creates multiple replicas in the same process


def create_deployment(model_name: str, model_path: str):
    """Create a deployment with unique name and autoscaling for a specific Kandinsky model"""
    
    # Create unique deployment name from model name
    deployment_name = f"kandinsky-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
    @serve.deployment(
        name=deployment_name,
        autoscaling_config={
            "min_replicas": 0,  # Scale to zero
            "max_replicas": 5,  # Max 5 replicas per model
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
                    # This ensures Ray can be imported during verification
                    "ray[serve]>=2.49.0",  # Match system Ray version for venv compatibility
                    # ML/AI framework dependencies
                    "numpy>=1.24.0",  # Required by torch and diffusers
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "diffusers>=0.24.0",  # Kandinsky 3 requires 0.24.0+
                    "accelerate>=0.20.0",
                    "transformers>=4.30.0",
                    # Image processing
                    "pillow>=9.0.0",
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
                }
            }
        }
    )
    class KandinskyEndpoint:
        """Kandinsky endpoint with autoscaling and dynamic input discovery"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path):
            self.model_loaded = False
            self.pipeline = None  # Main text-to-image pipeline
            self.img2img_pipeline = None  # Image-to-image pipeline
            self.model_name = model_name
            self.model_path = model_path
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            
            # Import torch and diffusers (they're in runtime_env, so should be available)
            # Don't delete and re-import - this causes Triton library registration conflicts
            try:
                import torch
                from diffusers import Kandinsky3Pipeline, AutoPipelineForText2Image
                print("✅ PyTorch and Diffusers available")
            except ImportError as import_err:
                print(f"⚠️  PyTorch/Diffusers not available: {import_err}")
                print("   Attempting to install...")
                try:
                    # Use the replica's Python (from runtime_env venv)
                    replica_python = sys.executable
                    print(f"   Using Python: {replica_python}")
                    result = subprocess.run(
                        [replica_python, "-m", "pip", "install", 
                         "diffusers>=0.24.0", "accelerate>=0.20.0", "torchvision>=0.15.0"],
                        timeout=600,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"pip install failed with return code {result.returncode}")
                    
                    # Import after installation (don't delete from sys.modules - causes Triton conflicts)
                    import torch
                    from diffusers import Kandinsky3Pipeline, AutoPipelineForText2Image
                    print("✅ Installed and imported PyTorch/Diffusers")
                except Exception as install_err:
                    print(f"❌ Installation failed: {install_err}")
                    raise ImportError(f"PyTorch/Diffusers not available and installation failed: {install_err}")
            
            # Load model in background thread
            def load_model():
                try:
                    # Import torch in the function scope to avoid UnboundLocalError
                    import torch
                    from diffusers import (
                        Kandinsky3Pipeline, 
                        AutoPipelineForText2Image,
                        AutoPipelineForImage2Image
                    )
                    
                    self.loading_started = True
                    print(f"🚀 Loading Kandinsky 3 model in background: {model_name}")
                    print(f"   HuggingFace Model ID: {model_path}")
                    
                    # Load text-to-image pipeline (main pipeline)
                    print(f"   Loading text-to-image pipeline from HuggingFace: {model_path}")
                    try:
                        # Try AutoPipelineForText2Image with variant="fp16" (known to work)
                        print("   Attempting to load text-to-image with AutoPipelineForText2Image...")
                        self.pipeline = AutoPipelineForText2Image.from_pretrained(
                            model_path,
                            variant="fp16",
                            torch_dtype=torch.float16
                        )
                        print("   ✅ Loaded text-to-image with AutoPipelineForText2Image (fp16 variant)")
                    except Exception as auto_err:
                        print(f"   ⚠️  Failed to load with AutoPipelineForText2Image: {auto_err}")
                        print("   Trying Kandinsky3Pipeline...")
                        try:
                            # Fallback to Kandinsky3Pipeline
                            self.pipeline = Kandinsky3Pipeline.from_pretrained(
                                model_path,
                                variant="fp16",
                                torch_dtype=torch.float16
                            )
                            print("   ✅ Loaded text-to-image with Kandinsky3Pipeline (fp16 variant)")
                        except Exception as kandinsky3_err:
                            print(f"   ⚠️  Failed to load with fp16 variant: {kandinsky3_err}")
                            print("   Trying without variant parameter...")
                            # Final fallback: try without variant
                            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16
                            )
                            print("   ✅ Loaded text-to-image without variant parameter")
                    
                    # For image-to-image and inpainting, reuse the base pipeline to save memory
                    # Kandinsky 3 pipelines are very large, so we can't load 3 separate copies
                    # Instead, we'll use the base pipeline for all capabilities
                    print("   Checking image-to-image capability...")
                    # Try to use AutoPipelineForImage2Image which can be created from the base pipeline
                    # or check if base pipeline supports image_to_image method
                    if hasattr(self.pipeline, 'image_to_image'):
                        print("   ✅ Base pipeline supports image_to_image method (reusing base pipeline)")
                        self.img2img_pipeline = self.pipeline  # Reuse base pipeline to save memory
                    else:
                        # Try to create img2img pipeline from base pipeline components (saves memory)
                        try:
                            print("   Attempting to create image-to-image pipeline from base components...")
                            # AutoPipelineForImage2Image can reuse components from text-to-image pipeline
                            self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
                                self.pipeline,
                                torch_dtype=torch.float16
                            )
                            print("   ✅ Created image-to-image pipeline from base (shares components)")
                        except Exception as img2img_err:
                            print(f"   ⚠️  Cannot create image-to-image from base: {img2img_err}")
                            print("   ⚠️  Image-to-image will not be available (memory constraints)")
                            self.img2img_pipeline = None
                    
                    
                    # Use CPU offloading instead of moving everything to GPU
                    # This saves memory by moving components to CPU when not in use
                    # Kandinsky 3 is too large to fit multiple copies on GPU
                    print("   Setting up CPU offloading to save GPU memory...")
                    
                    # Enable CPU offloading on all pipelines (moves components to CPU when idle)
                    if self.pipeline is not None and hasattr(self.pipeline, 'enable_model_cpu_offload'):
                        print("   Enabling CPU offloading on text-to-image pipeline...")
                        self.pipeline.enable_model_cpu_offload()
                        print("   ✅ CPU offloading enabled on text-to-image pipeline")
                    elif self.pipeline is not None and hasattr(self.pipeline, 'to'):
                        # Fallback: move to GPU if CPU offloading not available
                        print("   Moving text-to-image pipeline to GPU (CPU offloading not available)...")
                        self.pipeline = self.pipeline.to("cuda")
                        print("   ✅ text-to-image pipeline moved to GPU")
                    
                    # Handle image-to-image pipeline
                    if self.img2img_pipeline is not None:
                        if self.img2img_pipeline is self.pipeline:
                            print("   ✅ image-to-image uses base pipeline (CPU offloading already enabled)")
                        elif hasattr(self.img2img_pipeline, 'enable_model_cpu_offload'):
                            print("   Enabling CPU offloading on image-to-image pipeline...")
                            self.img2img_pipeline.enable_model_cpu_offload()
                            print("   ✅ CPU offloading enabled on image-to-image pipeline")
                        elif hasattr(self.img2img_pipeline, 'to'):
                            print("   ⚠️  Moving image-to-image to GPU (may cause OOM if separate pipeline)...")
                            try:
                                import torch
                                self.img2img_pipeline = self.img2img_pipeline.to("cuda")
                                print("   ✅ image-to-image pipeline moved to GPU")
                            except RuntimeError as oom_err:
                                if "out of memory" in str(oom_err).lower() or "OOM" in str(oom_err):
                                    print(f"   ❌ Out of memory loading image-to-image: {oom_err}")
                                    print("   ⚠️  Image-to-image will not be available")
                                    self.img2img_pipeline = None
                                else:
                                    raise
                    
                    
                    # Verify GPU usage
                    try:
                        import torch
                        if torch.cuda.is_available():
                            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
                            allocated = torch.cuda.memory_allocated(0) / 1024**3
                            reserved = torch.cuda.memory_reserved(0) / 1024**3
                            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            print(f"   GPU memory allocated: {allocated:.2f} GB / {total:.2f} GB")
                            print(f"   GPU memory reserved: {reserved:.2f} GB")
                            print(f"   GPU memory free: {total - reserved:.2f} GB")
                        else:
                            print("   ⚠️  CUDA not available - using CPU offloading")
                    except Exception as gpu_check_err:
                        print(f"   ⚠️  Could not verify GPU: {gpu_check_err}")
                    
                    # Enable attention slicing on all pipelines (saves memory)
                    for name, pipeline in [("text-to-image", self.pipeline), 
                                          ("image-to-image", self.img2img_pipeline)]:
                        if pipeline is not None and hasattr(pipeline, 'enable_attention_slicing'):
                            pipeline.enable_attention_slicing()
                            print(f"   ✅ Attention slicing enabled on {name} pipeline")
                    
                    # Discover schema after model is loaded (for main text-to-image pipeline)
                    self.schema = self._discover_pipeline_schema()
                    print(f"✅ Model loaded: {model_name}")
                    print(f"   Discovered {len(self.schema.get('inputs', {}).get('required', []))} required inputs")
                    print(f"   Discovered {len(self.schema.get('inputs', {}).get('optional', []))} optional inputs")
                    
                    # Report available capabilities
                    capabilities = []
                    if self.pipeline is not None:
                        capabilities.append("text-to-image")
                    if self.img2img_pipeline is not None:
                        capabilities.append("image-to-image")
                    print(f"   Available capabilities: {', '.join(capabilities)}")
                    
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
        
        def _discover_pipeline_schema(self) -> Dict[str, Any]:
            """Dynamically discover the pipeline's inputs and capabilities"""
            if self.pipeline is None:
                return {"error": "Pipeline not loaded"}
            
            schema = {
                "model_id": self.model_name,
                "pipeline_class": type(self.pipeline).__name__,
                "capabilities": [],
                "inputs": {
                    "required": [],
                    "optional": []
                },
                "methods": {},
                "available_pipelines": {}
            }
            
            # Discover capabilities by checking available pipelines and methods
            if self.pipeline is not None:
                schema["capabilities"].append("text-to-image")
                schema["available_pipelines"]["text-to-image"] = type(self.pipeline).__name__
            
            if self.img2img_pipeline is not None:
                schema["capabilities"].append("image-to-image")
                schema["available_pipelines"]["image-to-image"] = type(self.img2img_pipeline).__name__
                # Try to get image_to_image method signature
                pipeline_to_check = self.img2img_pipeline
                if hasattr(pipeline_to_check, 'image_to_image'):
                    try:
                        sig = inspect.signature(pipeline_to_check.image_to_image)
                        schema["methods"]["image_to_image"] = self._parse_signature(sig)
                    except:
                        pass
                elif hasattr(pipeline_to_check, '__call__'):
                    # If no image_to_image method, check __call__ signature
                    try:
                        sig = inspect.signature(pipeline_to_check.__call__)
                        schema["methods"]["image_to_image"] = self._parse_signature(sig)
                    except:
                        pass
            
            # Get the main __call__ signature (text-to-image)
            try:
                sig = inspect.signature(self.pipeline.__call__)
                call_schema = self._parse_signature(sig)
                schema["methods"]["__call__"] = call_schema
                
                # Separate required vs optional
                for param_name, param_info in call_schema["parameters"].items():
                    if param_info["required"]:
                        schema["inputs"]["required"].append(param_name)
                    else:
                        schema["inputs"]["optional"].append(param_name)
            except Exception as e:
                schema["error"] = f"Could not inspect signature: {e}"
            
            # Get default values
            schema["defaults"] = self._get_pipeline_defaults()
            
            return schema
        
        def _parse_signature(self, sig: inspect.Signature) -> Dict[str, Any]:
            """Parse a function signature into a schema"""
            params = {}
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter
                if param_name == 'self':
                    continue
                
                param_info = {
                    "name": param_name,
                    "required": param.default == inspect.Parameter.empty,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                }
                
                params[param_name] = param_info
            
            return {
                "parameters": params,
                "parameter_count": len(params)
            }
        
        def _get_pipeline_defaults(self) -> Dict[str, Any]:
            """Get default values that the pipeline uses"""
            defaults = {}
            
            # Standard defaults for Kandinsky pipelines
            defaults["num_inference_steps"] = 50
            defaults["guidance_scale"] = 7.5
            defaults["width"] = 512
            defaults["height"] = 512
            
            # Try to get from schema if available
            if self.schema and "methods" in self.schema:
                call_method = self.schema["methods"].get("__call__", {})
                params = call_method.get("parameters", {})
                for param_name, param_info in params.items():
                    if not param_info["required"] and param_info["default"] is not None:
                        defaults[param_name] = param_info["default"]
            
            return defaults
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            # For generation endpoints, check readiness immediately before any processing
            # This allows immediate response when scaling up from zero
            if path.endswith("/generate") or path.endswith("/text-to-image"):
                # Quick readiness check - return immediately if not ready
                if not (hasattr(self, 'model_loaded') and self.model_loaded and 
                        hasattr(self, 'pipeline') and self.pipeline is not None):
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60 seconds.",
                            "model": getattr(self, 'model_name', 'unknown'),
                            "estimated_ready_time_seconds": 60,
                            "retry_after": 60,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "60",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                return await self.handle_generate(request)
            elif path.endswith("/image-to-image") or path.endswith("/img2img"):
                # Quick readiness check for img2img
                if not (hasattr(self, 'model_loaded') and self.model_loaded and 
                        hasattr(self, 'img2img_pipeline') and self.img2img_pipeline is not None):
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60 seconds.",
                            "model": getattr(self, 'model_name', 'unknown'),
                            "estimated_ready_time_seconds": 60,
                            "retry_after": 60,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "60",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                return await self.handle_image_to_image(request)
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
            if not hasattr(self, 'pipeline') or self.pipeline is None:
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
                self.schema = self._discover_pipeline_schema()
            
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
            model_id = os.path.basename(self.model_path).replace(".safetensors", "").replace(".ckpt", "")
            return JSONResponse({
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "kandinsky"
                }]
            })
        
        async def handle_generate(self, request: Request) -> Any:
            """Handle text-to-image generation requests using discovered schema"""
            try:
                # Check if model is ready
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'pipeline') and 
                    self.pipeline is not None
                )
                
                if not model_ready:
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60 seconds.",
                            "model": self.model_name,
                            "estimated_ready_time_seconds": 60,
                            "retry_after": 60,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "60",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                
                query_start_time = time.time()
                
                # Get request data
                data = await request.json()
                
                # Use discovered schema to build pipeline kwargs dynamically
                pipeline_kwargs = self._build_pipeline_kwargs(data)
                
                # Ensure torch is available (it should be from __init__, but check just in case)
                try:
                    import torch
                except ImportError:
                    return JSONResponse(
                        {"error": "PyTorch is not available. Please install torch."},
                        status_code=500
                    )
                
                # Generate image
                with torch.inference_mode():
                    result = self.pipeline(**pipeline_kwargs)
                
                # Handle diffusers pipeline return format (object with .images attribute)
                if hasattr(result, 'images'):
                    image = result.images[0]
                else:
                    raise ValueError(f"Unexpected result type from pipeline: {type(result)}")
                
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                metrics_end = self._get_resource_metrics()
                
                # Convert image to base64
                from PIL import Image
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data = f"data:image/png;base64,{img_base64}"
                
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
                
                # Clean pipeline_kwargs for JSON serialization (remove PIL Images, generators, etc.)
                serializable_kwargs = {}
                for key, value in pipeline_kwargs.items():
                    # Skip non-serializable objects
                    if isinstance(value, (type(None), str, int, float, bool, list, dict)):
                        serializable_kwargs[key] = value
                    elif hasattr(value, '__class__'):
                        # For complex objects, just store the type name
                        class_name = value.__class__.__name__
                        if 'Image' in class_name:
                            serializable_kwargs[key] = f"<{class_name} object>"
                        elif 'Generator' in class_name:
                            serializable_kwargs[key] = f"<{class_name} object>"
                        else:
                            serializable_kwargs[key] = f"<{class_name} object>"
                    else:
                        serializable_kwargs[key] = str(value)
                
                return JSONResponse({
                    "status": "success",
                    "model": self.model_name,
                    "image": image_data,
                    "parameters": serializable_kwargs,
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
        async def handle_image_to_image(self, request: Request) -> Any:
            """Handle image-to-image generation requests"""
            try:
                print("🖼️  Image-to-image request received")
                # Check if model is ready
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'img2img_pipeline') and 
                    self.img2img_pipeline is not None
                )
                
                if not model_ready:
                    print("⚠️  Image-to-image pipeline not ready")
                    return JSONResponse(
                        {
                            "status": "scaling_up",
                            "message": "The model is currently scaled to zero and is scaling up. Please poll this endpoint or retry in ~60 seconds.",
                            "model": self.model_name,
                            "estimated_ready_time_seconds": 60,
                            "retry_after": 60,
                            "poll_url": str(request.url),
                            "action": "retry_later"
                        },
                        status_code=202,
                        headers={
                            "Retry-After": "60",
                            "X-Status": "scaling-up",
                            "X-Action": "retry-later",
                            "Content-Type": "application/json"
                        }
                    )
                
                print("✅ Image-to-image pipeline is ready")
                query_start_time = time.time()
                
                # Get request data
                print("📥 Parsing request JSON...")
                try:
                    data = await request.json()
                    print(f"✅ Request parsed. Keys: {list(data.keys())}")
                except Exception as json_err:
                    print(f"❌ Failed to parse JSON: {json_err}")
                    import traceback
                    traceback.print_exc()
                    return JSONResponse(
                        {"error": f"Invalid JSON: {str(json_err)}"},
                        status_code=400
                    )
                
                # Ensure we have an image
                if "image" not in data:
                    print("❌ Missing 'image' parameter")
                    return JSONResponse(
                        {"error": "Missing required parameter: 'image' (base64 encoded image data)"},
                        status_code=400
                    )
                
                print("🖼️  Image parameter found, decoding...")
                try:
                    # Decode image early to catch errors
                    image_obj = self._decode_base64_image(data["image"])
                    print(f"✅ Image decoded. Size: {image_obj.size}, Mode: {image_obj.mode}")
                except Exception as img_err:
                    print(f"❌ Failed to decode image: {img_err}")
                    import traceback
                    traceback.print_exc()
                    return JSONResponse(
                        {"error": f"Failed to decode image: {str(img_err)}"},
                        status_code=400
                    )
                
                # Build pipeline kwargs using the img2img pipeline schema
                print("🔧 Building pipeline kwargs...")
                original_pipeline = self.pipeline
                self.pipeline = self.img2img_pipeline
                try:
                    # Discover img2img schema if not already done
                    print("📋 Discovering img2img schema...")
                    img2img_schema = self._discover_pipeline_schema()
                    original_schema = self.schema
                    self.schema = img2img_schema
                    print(f"✅ Schema discovered. Available parameters: {list(img2img_schema.get('inputs', {}).get('optional', []))}")
                    
                    pipeline_kwargs = self._build_pipeline_kwargs(data)
                    print(f"✅ Pipeline kwargs built. Keys: {list(pipeline_kwargs.keys())}")
                    print(f"   Parameters: {', '.join([f'{k}={type(v).__name__}' for k, v in pipeline_kwargs.items()])}")
                    self.schema = original_schema
                except Exception as kwargs_err:
                    print(f"❌ Failed to build pipeline kwargs: {kwargs_err}")
                    import traceback
                    traceback.print_exc()
                    raise
                finally:
                    self.pipeline = original_pipeline
                
                # Ensure torch is available
                try:
                    import torch
                    print("✅ PyTorch available")
                except ImportError:
                    print("❌ PyTorch not available")
                    return JSONResponse(
                        {"error": "PyTorch is not available. Please install torch."},
                        status_code=500
                    )
                
                # Generate image
                print("🎨 Starting image generation...")
                print(f"   Pipeline type: {type(self.img2img_pipeline).__name__}")
                print(f"   Has image_to_image method: {hasattr(self.img2img_pipeline, 'image_to_image')}")
                print(f"   Pipeline kwargs: {list(pipeline_kwargs.keys())}")
                
                try:
                    with torch.inference_mode():
                        # Use image_to_image method if available, otherwise use __call__ with image parameter
                        if hasattr(self.img2img_pipeline, 'image_to_image'):
                            print("   Using image_to_image() method...")
                            result = self.img2img_pipeline.image_to_image(**pipeline_kwargs)
                        else:
                            print("   Using __call__() method...")
                            result = self.img2img_pipeline(**pipeline_kwargs)
                    print(f"✅ Image generation completed. Result type: {type(result)}")
                    print(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                except Exception as gen_err:
                    print(f"❌ Image generation failed: {gen_err}")
                    import traceback
                    traceback.print_exc()
                    return JSONResponse(
                        {
                            "error": f"Image generation failed: {str(gen_err)}",
                            "error_type": type(gen_err).__name__,
                            "pipeline_type": type(self.img2img_pipeline).__name__,
                            "pipeline_kwargs_keys": list(pipeline_kwargs.keys())
                        },
                        status_code=500
                    )
                
                # Handle diffusers pipeline return format
                print("📦 Processing pipeline result...")
                print(f"   Result type: {type(result)}")
                print(f"   Has 'images' attribute: {hasattr(result, 'images')}")
                
                try:
                    if hasattr(result, 'images'):
                        print(f"   Result.images type: {type(result.images)}")
                        print(f"   Result.images length: {len(result.images) if hasattr(result.images, '__len__') else 'N/A'}")
                        image = result.images[0]
                        print(f"✅ Image extracted. Type: {type(image)}")
                        if hasattr(image, 'size'):
                            print(f"   Image size: {image.size}")
                        if hasattr(image, 'mode'):
                            print(f"   Image mode: {image.mode}")
                    else:
                        error_msg = f"Unexpected result type from pipeline: {type(result)}"
                        print(f"❌ {error_msg}")
                        print(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                        # Try to get more info about the result
                        if hasattr(result, '__dict__'):
                            print(f"   Result __dict__: {result.__dict__}")
                        raise ValueError(error_msg)
                except Exception as result_err:
                    print(f"❌ Failed to process result: {result_err}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                query_end_time = time.time()
                query_duration = query_end_time - query_start_time
                metrics_end = self._get_resource_metrics()
                
                # Convert image to base64
                print("🔄 Converting image to base64...")
                try:
                    from PIL import Image
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data = f"data:image/png;base64,{img_base64}"
                    print(f"✅ Image converted. Base64 length: {len(img_base64)}")
                except Exception as conv_err:
                    print(f"❌ Failed to convert image: {conv_err}")
                    import traceback
                    traceback.print_exc()
                    raise
                
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
                
                # Clean pipeline_kwargs for JSON serialization (remove PIL Images, generators, etc.)
                serializable_kwargs = {}
                for key, value in pipeline_kwargs.items():
                    # Skip non-serializable objects
                    if isinstance(value, (type(None), str, int, float, bool, list, dict)):
                        serializable_kwargs[key] = value
                    elif hasattr(value, '__class__'):
                        # For complex objects, just store the type name
                        class_name = value.__class__.__name__
                        if 'Image' in class_name:
                            serializable_kwargs[key] = f"<{class_name} object>"
                        elif 'Generator' in class_name:
                            serializable_kwargs[key] = f"<{class_name} object>"
                        else:
                            serializable_kwargs[key] = f"<{class_name} object>"
                    else:
                        serializable_kwargs[key] = str(value)
                
                print(f"✅ Image-to-image request completed successfully in {query_duration:.2f}s")
                return JSONResponse({
                    "status": "success",
                    "model": self.model_name,
                    "capability": "image-to-image",
                    "image": image_data,
                    "parameters": serializable_kwargs,
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"❌ Image-to-image request failed: {error_type}: {error_msg}")
                import traceback
                traceback.print_exc()
                return JSONResponse(
                    {
                        "error": error_msg,
                        "error_type": error_type,
                        "traceback": traceback.format_exc()
                    },
                    status_code=500
                )
        
        def _build_pipeline_kwargs(self, data: Dict) -> Dict:
            """Build pipeline kwargs from request data using discovered schema"""
            import torch  # Import torch here since it's needed for Generator
            kwargs = {}
            
            # Get defaults from schema
            defaults = self.schema.get("defaults", {}) if self.schema else {}
            
            # Get required inputs from schema
            required_inputs = self.schema.get("inputs", {}).get("required", []) if self.schema else []
            optional_inputs = self.schema.get("inputs", {}).get("optional", []) if self.schema else []
            
            # Process all known inputs from schema
            all_inputs = required_inputs + optional_inputs
            
            for param_name in all_inputs:
                # Get value from request data
                value = data.get(param_name)
                
                # If not provided, use default from schema
                if value is None:
                    # Check schema defaults
                    if param_name in defaults:
                        value = defaults[param_name]
                    # Check method parameter defaults
                    elif self.schema and "methods" in self.schema:
                        call_method = self.schema["methods"].get("__call__", {})
                        params = call_method.get("parameters", {})
                        if param_name in params:
                            param_info = params[param_name]
                            if not param_info["required"] and param_info["default"] is not None:
                                value = param_info["default"]
                
                # Only add if we have a value (or it's required)
                if value is not None or param_name in required_inputs:
                    # Handle special cases
                    if param_name == "generator" and isinstance(value, int):
                        # Convert seed to generator
                        generator = torch.Generator(device="cuda").manual_seed(value)
                        kwargs[param_name] = generator
                    elif param_name == "seed" and value is not None:
                        # Convert seed to generator
                        generator = torch.Generator(device="cuda").manual_seed(value)
                        kwargs["generator"] = generator
                    elif param_name in ["image", "mask_image", "control_image"] and isinstance(value, str):
                        # Decode base64 image
                        kwargs[param_name] = self._decode_base64_image(value)
                    else:
                        kwargs[param_name] = value
            
            # Handle seed -> generator conversion (if seed is provided but generator is expected)
            # This is a type conversion, not a parameter name remapping
            if "seed" in data and "generator" in all_inputs and "generator" not in kwargs:
                seed = data.get("seed")
                if seed is not None:
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    kwargs["generator"] = generator
            
            return kwargs
        
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
    
    return KandinskyEndpoint


def create_app(model_name: str, model_path: str):
    """Create a Ray Serve application for a specific Kandinsky model"""
    deployment_class = create_deployment(model_name, model_path)
    return deployment_class.bind(model_name=model_name, model_path=model_path)
