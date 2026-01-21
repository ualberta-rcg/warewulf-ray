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
            "max_replicas": 1,  # Max 1 replica per model
            "target_num_ongoing_requests_per_replica": 1,
            "initial_replicas": 0,
            "downscale_delay_s": 300,  # Wait 5 minutes before scaling down
            "upscale_delay_s": 0,  # Scale up immediately
        },
        ray_actor_options={
            "num_gpus": 1,
            "runtime_env": {
                "pip": [
                    # Ray and core dependencies first (ensures venv has everything needed)
                    "ray[serve]>=2.49.0",  # Includes Ray Serve and all core dependencies (packaging, setuptools, etc.)
                    "setuptools>=65.0",  # Required for building packages
                    "wheel>=0.38.0",  # Required for building packages
                    "packaging>=21.0",  # Required by Ray for runtime env verification
                    # ML/AI framework dependencies
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "diffusers>=0.21.0",
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
                    # Note: bezier is installed manually with --no-build-isolation in __init__
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
            import sys
            import subprocess
            
            self.model_loaded = False
            self.pipeline = None
            self.model_name = model_name
            self.model_path = model_path
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            
            # Install bezier with --no-build-isolation and BEZIER_INSTALL_PREFIX (required for proper build)
            try:
                import bezier
                print("✅ bezier already available")
            except ImportError:
                print("⚠️  bezier not available. Installing with --no-build-isolation...")
                try:
                    replica_python = sys.executable
                    # Set BEZIER_INSTALL_PREFIX to a temporary directory
                    import tempfile
                    bezier_prefix = tempfile.mkdtemp(prefix="bezier_install_")
                    env = os.environ.copy()
                    env["BEZIER_INSTALL_PREFIX"] = bezier_prefix
                    
                    result = subprocess.run(
                        [replica_python, "-m", "pip", "install", "--no-build-isolation", "bezier>=2021.2.12"],
                        env=env,
                        timeout=300,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print("✅ bezier installed successfully with --no-build-isolation")
                        # Reload module
                        if 'bezier' in sys.modules:
                            del sys.modules['bezier']
                        import bezier
                    else:
                        print(f"⚠️  bezier installation failed (return code: {result.returncode})")
                        if result.stderr:
                            print(f"   stderr: {result.stderr[:500]}")
                        if result.stdout:
                            print(f"   stdout: {result.stdout[:500]}")
                        raise RuntimeError(f"Failed to install bezier: {result.stderr}")
                except Exception as bezier_err:
                    print(f"⚠️  Could not install bezier: {bezier_err}")
                    # Don't fail completely - bezier might not be critical for all models
                    import traceback
                    traceback.print_exc()
            
            # Import torch and diffusers (they're in runtime_env, so should be available)
            # Don't delete and re-import - this causes Triton library registration conflicts
            try:
                import torch
                from diffusers import Kandinsky3Pipeline
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
                         "diffusers>=0.21.0", "accelerate>=0.20.0", "torchvision>=0.15.0"],
                        timeout=600,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"pip install failed with return code {result.returncode}")
                    
                    # Import after installation (don't delete from sys.modules - causes Triton conflicts)
                    import torch
                    from diffusers import Kandinsky3Pipeline
                    print("✅ Installed and imported PyTorch/Diffusers")
                except Exception as install_err:
                    print(f"❌ Installation failed: {install_err}")
                    raise ImportError(f"PyTorch/Diffusers not available and installation failed: {install_err}")
            
            # Load model in background thread
            def load_model():
                try:
                    # Import torch in the function scope to avoid UnboundLocalError
                    import torch
                    from diffusers import Kandinsky3Pipeline
                    
                    self.loading_started = True
                    print(f"🚀 Loading Kandinsky 3 model in background: {model_name}")
                    print(f"   HuggingFace Model ID: {model_path}")
                    
                    # Load from HuggingFace using diffusers (model_path is always a HuggingFace model ID)
                    print(f"   Loading from HuggingFace: {model_path}")
                    self.pipeline = Kandinsky3Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16
                    )
                    
                    # Move to GPU
                    if hasattr(self.pipeline, 'to'):
                        print("   Moving pipeline to GPU...")
                        self.pipeline = self.pipeline.to("cuda")
                        print("   ✅ Pipeline moved to GPU")
                    else:
                        print("   ⚠️  Pipeline doesn't have .to() method (may already be on GPU)")
                    
                    # Verify GPU usage
                    try:
                        import torch
                        if torch.cuda.is_available():
                            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
                            print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                        else:
                            print("   ⚠️  CUDA not available - model may be on CPU")
                    except Exception as gpu_check_err:
                        print(f"   ⚠️  Could not verify GPU: {gpu_check_err}")
                    
                    # Enable attention slicing if available (saves memory)
                    if hasattr(self.pipeline, 'enable_attention_slicing'):
                        self.pipeline.enable_attention_slicing()
                        print("   ✅ Attention slicing enabled")
                    else:
                        print("   ⚠️  Pipeline doesn't support attention slicing")
                    
                    # Discover schema after model is loaded
                    self.schema = self._discover_pipeline_schema()
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
                "methods": {}
            }
            
            # Discover capabilities by checking available methods
            if hasattr(self.pipeline, '__call__'):
                schema["capabilities"].append("text-to-image")
            
            if hasattr(self.pipeline, 'image_to_image'):
                schema["capabilities"].append("image-to-image")
                try:
                    sig = inspect.signature(self.pipeline.image_to_image)
                    schema["methods"]["image_to_image"] = self._parse_signature(sig)
                except:
                    pass
            
            if hasattr(self.pipeline, 'inpaint'):
                schema["capabilities"].append("inpainting")
                try:
                    sig = inspect.signature(self.pipeline.inpaint)
                    schema["methods"]["inpaint"] = self._parse_signature(sig)
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
            
            if path.endswith("/generate") or path.endswith("/text-to-image"):
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
                
                return JSONResponse({
                    "status": "success",
                    "model": self.model_name,
                    "image": image_data,
                    "parameters": pipeline_kwargs,
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
        def _build_pipeline_kwargs(self, data: Dict) -> Dict:
            """Build pipeline kwargs from request data using discovered schema"""
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
            
            # Handle legacy/fallback parameters not in schema
            # (for backwards compatibility and parameter name mapping)
            
            # Map common parameter names to pipeline-specific names
            # kandinsky3 uses "text" instead of "prompt", "steps" instead of "num_inference_steps"
            if (kwargs.get("text") is None or "text" not in kwargs) and "prompt" in data:
                # Map "prompt" -> "text" for kandinsky3
                if "text" in all_inputs:
                    kwargs["text"] = data["prompt"]
                elif "prompt" in all_inputs:
                    kwargs["prompt"] = data["prompt"]
            
            if "negative_text" not in kwargs and "negative_prompt" in data:
                # Map "negative_prompt" -> "negative_text" for kandinsky3
                if "negative_text" in all_inputs:
                    kwargs["negative_text"] = data.get("negative_prompt")
                elif "negative_prompt" in all_inputs:
                    kwargs["negative_prompt"] = data.get("negative_prompt")
            
            # Map "num_inference_steps" -> "steps" for kandinsky3
            if "steps" not in kwargs and "num_inference_steps" in data:
                if "steps" in all_inputs:
                    kwargs["steps"] = data.get("num_inference_steps")
                elif "num_inference_steps" in all_inputs:
                    kwargs["num_inference_steps"] = data.get("num_inference_steps")
            
            # Only add fallback defaults if the parameter exists in the schema
            # Don't add parameters that don't exist in the discovered schema
            if "guidance_scale" in all_inputs and "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = data.get("guidance_scale", defaults.get("guidance_scale", 7.5))
            
            if "width" in all_inputs and "width" not in kwargs:
                kwargs["width"] = data.get("width", defaults.get("width", 512))
            
            if "height" in all_inputs and "height" not in kwargs:
                kwargs["height"] = data.get("height", defaults.get("height", 512))
            
            if "seed" in data and "generator" not in kwargs:
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
