"""
Ray Serve deployment for Kandinsky models
Version 2: Separate deployments per model with autoscaling and dynamic input discovery
"""

import os
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

# Try to import torch separately (needed even if diffusers import fails)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

# Try to import Kandinsky pipelines
try:
    from diffusers import KandinskyV22Pipeline
    KANDINSKY_V22_AVAILABLE = True
except ImportError:
    KANDINSKY_V22_AVAILABLE = False

# Try to import Kandinsky 3 (may not be available in all diffusers versions)
try:
    from diffusers import Kandinsky3Pipeline
    KANDINSKY_V3_AVAILABLE = True
except ImportError:
    KANDINSKY_V3_AVAILABLE = False

KANDINSKY_AVAILABLE = KANDINSKY_V22_AVAILABLE or KANDINSKY_V3_AVAILABLE


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
                    "diffusers>=0.21.0",
                    "accelerate>=0.20.0",
                    "transformers>=4.30.0",
                    "torch>=2.0.0",
                    "torchvision>=0.15.0",
                    "pillow>=9.0.0",
                    "fastapi>=0.100.0",
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0"
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
            self.is_v3 = False
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            
            # Check if Kandinsky is available
            if not KANDINSKY_AVAILABLE:
                print("⚠️  Kandinsky pipelines not available. Attempting to install...")
                try:
                    ray_python = "/opt/ray/bin/python"
                    if os.path.exists(ray_python):
                        print("   Installing diffusers and dependencies...")
                        result = subprocess.run(
                            [ray_python, "-m", "pip", "install", 
                             "diffusers>=0.21.0", "accelerate>=0.20.0", "torchvision>=0.15.0"],
                            timeout=600
                        )
                        if result.returncode != 0:
                            raise RuntimeError(f"pip install failed with return code {result.returncode}")
                        
                        # Reload modules
                        import importlib
                        for mod in ['diffusers', 'torch', 'PIL']:
                            if mod in sys.modules:
                                del sys.modules[mod]
                        import torch
                        from diffusers import KandinskyV22Pipeline
                        # Update global availability flags
                        global KANDINSKY_V22_AVAILABLE, KANDINSKY_V3_AVAILABLE, KANDINSKY_AVAILABLE
                        KANDINSKY_V22_AVAILABLE = True
                        try:
                            from diffusers import Kandinsky3Pipeline
                            KANDINSKY_V3_AVAILABLE = True
                            print("✅ Kandinsky pipelines installed successfully (V2.2 and V3)")
                        except ImportError:
                            KANDINSKY_V3_AVAILABLE = False
                            print("✅ Kandinsky V2.2 installed successfully (V3 not available in this diffusers version)")
                        KANDINSKY_AVAILABLE = True
                    else:
                        raise ImportError("Kandinsky not available and cannot install")
                except Exception as e:
                    print(f"❌ Failed to install Kandinsky: {e}")
                    raise ImportError(f"Kandinsky not available and installation failed: {e}")
            
            # Load model in background thread
            def load_model():
                try:
                    self.loading_started = True
                    print(f"🚀 Loading Kandinsky model in background: {model_name}")
                    print(f"   Model path: {model_path}")
                    print(f"   Path exists: {os.path.exists(model_path) if model_path else 'N/A'}")
                    if model_path and os.path.exists(model_path):
                        print(f"   Is directory: {os.path.isdir(model_path)}")
                        print(f"   Is file: {os.path.isfile(model_path)}")
                    
                    # Detect if it's V3 or V2.2 based on filename
                    model_lower = model_path.lower() if model_path else ""
                    is_v3 = ("kandinsky3" in model_lower or "v3" in model_lower) and KANDINSKY_V3_AVAILABLE
                    
                    if is_v3 and not KANDINSKY_V3_AVAILABLE:
                        print("⚠️  Kandinsky V3 requested but not available in diffusers")
                        print("   Falling back to Kandinsky V2.2")
                        is_v3 = False
                    
                    # Check if path is a directory or a file
                    if model_path and os.path.isdir(model_path):
                        # Load from directory (HuggingFace format)
                        print("   Loading from directory (HuggingFace format)")
                        if is_v3 and KANDINSKY_V3_AVAILABLE:
                            print("   Detected Kandinsky V3 - using Kandinsky3Pipeline")
                            try:
                                self.pipeline = Kandinsky3Pipeline.from_pretrained(
                                    model_path,
                                    torch_dtype=torch.float16,
                                    local_files_only=True
                                )
                                self.is_v3 = True
                            except Exception as e:
                                print(f"   ⚠️  Failed to load as Kandinsky V3: {e}")
                                print("   Falling back to Kandinsky V2.2")
                                is_v3 = False
                                self.pipeline = KandinskyV22Pipeline.from_pretrained(
                                    model_path,
                                    torch_dtype=torch.float16,
                                    local_files_only=True
                                )
                                self.is_v3 = False
                        else:
                            print("   Using KandinskyV22Pipeline (V2.2)")
                            self.pipeline = KandinskyV22Pipeline.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                local_files_only=True
                            )
                            self.is_v3 = False
                    elif model_path and os.path.isfile(model_path):
                        # Try to load from single file (checkpoint)
                        print("   Loading from single checkpoint file")
                        print("   ⚠️  Note: Kandinsky pipelines typically don't support single checkpoint files")
                        print("   Consider using a HuggingFace model directory or model ID instead")
                        is_safetensors = model_path.endswith(".safetensors")
                        
                        # Try from_single_file first (if supported)
                        try:
                            if is_v3 and KANDINSKY_V3_AVAILABLE:
                                print("   Detected Kandinsky V3 - trying Kandinsky3Pipeline")
                                if hasattr(Kandinsky3Pipeline, 'from_single_file'):
                                    self.pipeline = Kandinsky3Pipeline.from_single_file(
                                        model_path,
                                        torch_dtype=torch.float16,
                                        use_safetensors=is_safetensors
                                    )
                                    self.is_v3 = True
                                else:
                                    raise AttributeError("from_single_file not available for Kandinsky3Pipeline")
                            else:
                                print("   Using KandinskyV22Pipeline (V2.2)")
                                if hasattr(KandinskyV22Pipeline, 'from_single_file'):
                                    self.pipeline = KandinskyV22Pipeline.from_single_file(
                                        model_path,
                                        torch_dtype=torch.float16,
                                        use_safetensors=is_safetensors
                                    )
                                    self.is_v3 = False
                                else:
                                    raise AttributeError("from_single_file not available for KandinskyV22Pipeline")
                        except (AttributeError, TypeError, Exception) as e:
                            # from_single_file not supported or failed
                            error_msg = str(e)
                            print(f"   ❌ from_single_file failed: {error_msg}")
                            print("   Kandinsky pipelines do not support loading from single checkpoint files.")
                            print("   Please use one of the following:")
                            print("   1. A HuggingFace model directory (with config.json, etc.)")
                            print("   2. A HuggingFace model ID (e.g., 'ai-forever/kandinsky-2.2-decoder')")
                            print("   3. Convert the checkpoint to a directory structure")
                            raise RuntimeError(
                                f"Failed to load Kandinsky model from checkpoint file: {model_path}\n"
                                f"Error: {error_msg}\n"
                                f"Kandinsky pipelines require a HuggingFace model directory or model ID, "
                                f"not a single checkpoint file. Please convert the checkpoint or use a model directory."
                            )
                    else:
                        # Try as HuggingFace model ID
                        print(f"   Treating as HuggingFace model ID: {model_path}")
                        if is_v3 and KANDINSKY_V3_AVAILABLE:
                            print("   Detected Kandinsky V3 - trying Kandinsky3Pipeline")
                            try:
                                self.pipeline = Kandinsky3Pipeline.from_pretrained(
                                    model_path,
                                    torch_dtype=torch.float16
                                )
                                self.is_v3 = True
                            except Exception as e:
                                print(f"   ⚠️  Failed to load as Kandinsky V3: {e}")
                                print("   Falling back to Kandinsky V2.2")
                                is_v3 = False
                                self.pipeline = KandinskyV22Pipeline.from_pretrained(
                                    model_path,
                                    torch_dtype=torch.float16
                                )
                                self.is_v3 = False
                        else:
                            print("   Using KandinskyV22Pipeline (V2.2)")
                            self.pipeline = KandinskyV22Pipeline.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16
                            )
                            self.is_v3 = False
                    
                    # Move to GPU
                    self.pipeline = self.pipeline.to("cuda")
                    self.pipeline.enable_attention_slicing()
                    
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
                with torch.inference_mode():
                    result = self.pipeline(**pipeline_kwargs)
                
                image = result.images[0]
                
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
            # (for backwards compatibility)
            if "prompt" not in kwargs and "prompt" in data:
                kwargs["prompt"] = data["prompt"]
            if "negative_prompt" not in kwargs and "negative_prompt" in data:
                kwargs["negative_prompt"] = data.get("negative_prompt")
            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = data.get("num_inference_steps", defaults.get("num_inference_steps", 50))
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = data.get("guidance_scale", defaults.get("guidance_scale", 7.5))
            if "width" not in kwargs:
                kwargs["width"] = data.get("width", defaults.get("width", 512))
            if "height" not in kwargs:
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
