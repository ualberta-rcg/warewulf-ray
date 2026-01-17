"""
Ray Serve deployment for GraphCast weather forecasting models
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
import json
from typing import Any, Dict, List, Optional
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve

# Try to import torch separately
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

# Try to import GraphCast dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  NumPy not available. Install with: pip install numpy")


def create_deployment(model_name: str, model_path: str):
    """Create a deployment with unique name and autoscaling for a specific GraphCast model"""
    
    # Create unique deployment name from model name
    deployment_name = f"graphcast-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
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
                    "transformers>=4.30.0",
                    "accelerate>=0.20.0",
                    "torch>=2.0.0",
                    "numpy>=1.24.0",
                    "fastapi>=0.100.0",
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0",
                    "xarray>=2023.1.0",  # For weather data handling
                    "netcdf4>=1.6.0",  # For NetCDF files (common in weather data)
                    # GraphCast dependencies (JAX-based)
                    "jax[cuda12]>=0.4.0",  # JAX with CUDA support (adjust version as needed)
                    "jaxlib>=0.4.0",
                    "dm-haiku>=0.0.9",  # DeepMind Haiku (needed by GraphCast)
                    "dm-tree>=0.1.8",  # DeepMind Tree
                    "huggingface-hub>=0.17.0",  # For downloading from HuggingFace
                    # Additional GraphCast dependencies (from setup.py)
                    "chex>=0.1.0",  # Testing utilities for JAX
                    "jraph",  # Graph neural networks for JAX (use any available version, including dev)
                    "matplotlib>=3.5.0",  # Plotting
                    "pandas>=1.3.0",  # Data manipulation
                    "scipy>=1.7.0",  # Scientific computing
                    "typing_extensions>=4.0.0",  # Type hints
                    "trimesh>=3.0.0",  # 3D mesh processing (required by GraphCast)
                    "rtree>=1.0.0",  # Spatial indexing (required by GraphCast)
                    "cartopy>=0.21.0",  # Cartographic projections (optional but used by GraphCast)
                    "dask>=2023.1.0",  # Parallel computing (optional but used by GraphCast)
                    # Note: GraphCast library is loaded from /data/models/graphcast (NFS mount)
                    # Run: bash ray-graphcast-endpoints/setup_graphcast.sh to clone it
                ],
                "env_vars": {
                    "HF_HOME": "/data/models",
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                    "GRAPHCAST_PATH": "/data/models/graphcast",  # Path for GraphCast repo
                }
            }
        }
    )
    class GraphCastEndpoint:
        """GraphCast weather forecasting endpoint with autoscaling and dynamic input discovery"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path):
            import sys
            import subprocess
            
            self.model_loaded = False
            self.model = None
            self.tokenizer = None
            self.model_name = model_name
            self.model_path = model_path
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            # GraphCast-specific attributes
            self.checkpoint = None
            self.params = None
            self.model_config = None
            self.task_config = None
            self.is_graphcast = False  # Track if using GraphCast (JAX) vs PyTorch model
            
            # Start loading in background thread (non-blocking)
            loading_thread = threading.Thread(target=self._load_model, daemon=True)
            loading_thread.start()
        
        def _load_model(self):
            """Load the GraphCast model in background"""
            try:
                # Import torch in the function scope to avoid UnboundLocalError
                import torch
                
                self.loading_started = True
                print(f"🚀 Loading GraphCast model in background: {model_name}")
                print(f"   Model path: {model_path}")
                
                # GraphCast models can come from various sources - try multiple approaches
                print(f"   Attempting to load GraphCast model: {model_path}")
                
                model_loaded = False
                
                # 1. Check if it's a local directory or file
                if model_path and os.path.isdir(model_path):
                    print("   Loading from local directory...")
                    try:
                        from transformers import AutoModel
                        self.model = AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            local_files_only=True,
                            trust_remote_code=True
                        )
                        model_loaded = True
                        print("   ✅ Loaded from local directory")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load from directory: {e}")
                
                # 2. Try Google's GraphCast library (uses JAX, loads from HuggingFace)
                if not model_loaded:
                    print("   Attempting to use Google's GraphCast library (JAX-based)...")
                    graphcast_loading_failed = False
                    try:
                        # Try importing GraphCast libraries
                        graphcast_available = False
                        
                        # First, try to use GraphCast from NFS mount (if manually cloned)
                        graphcast_path = os.environ.get("GRAPHCAST_PATH", "/data/models/graphcast")
                        graphcast_module_path = os.path.join(graphcast_path, "graphcast")
                        
                        if os.path.exists(graphcast_module_path):
                            print(f"   Found GraphCast repository at: {graphcast_path}")
                            # Add to Python path
                            import sys
                            if graphcast_path not in sys.path:
                                sys.path.insert(0, graphcast_path)
                                print(f"   Added {graphcast_path} to Python path")
                        
                        try:
                            from graphcast import checkpoint
                            from graphcast import graphcast as gc_module
                            graphcast_available = True
                            print("   ✅ GraphCast library imported successfully")
                        except ImportError as import_err:
                            error_details = str(import_err)
                            print(f"   ❌ GraphCast library import failed: {error_details}")
                            import traceback
                            print("   Import traceback:")
                            traceback.print_exc()
                            
                            # Check if it's a missing dependency
                            if "No module named" in error_details:
                                missing_module = error_details.split("'")[1] if "'" in error_details else "unknown"
                                print(f"   💡 Missing dependency: {missing_module}")
                                print(f"   Install with: pip install {missing_module}")
                            
                            print(f"   💡 Ensure GraphCast is cloned to: {graphcast_path}")
                            print(f"   Run: bash ray-graphcast-endpoints/setup_graphcast.sh")
                            # Don't raise - let it fall through to show all errors
                            graphcast_available = False
                        
                        if not graphcast_available:
                            print("   ⚠️  Skipping GraphCast loading - library not available")
                            # Don't raise here - continue to try HuggingFace transformers
                        
                        if not graphcast_available:
                            # Skip the rest if GraphCast library isn't available
                            print("   ⚠️  Cannot proceed with GraphCast loading - library not imported")
                            raise ImportError("GraphCast library not available - skipping GraphCast loading")
                        
                        try:
                            from huggingface_hub import hf_hub_download, list_repo_files
                            print("   ✅ HuggingFace Hub available")
                        except ImportError as hf_import_err:
                            print(f"   ⚠️  huggingface_hub not available: {hf_import_err}")
                            raise ImportError(f"huggingface_hub not available: {hf_import_err}")
                        
                        try:
                            import jax
                            print(f"   ✅ JAX available (version: {jax.__version__})")
                        except ImportError as jax_import_err:
                            print(f"   ⚠️  JAX not available: {jax_import_err}")
                            raise ImportError(f"JAX not available: {jax_import_err}")
                        
                        print("   ✅ All GraphCast dependencies available")
                        
                        # GraphCast models on HuggingFace
                        repo_ids_to_try = [
                            model_path if "/" in model_path and model_path != "google-deepmind/graphcast" else None,  # User provided HF ID (skip invalid)
                            "shermansiu/dm_graphcast",  # Full resolution
                            "shermansiu/dm_graphcast_small",  # Smaller version
                        ]
                        
                        # Filter out None values
                        repo_ids_to_try = [r for r in repo_ids_to_try if r]
                        
                        for repo_id in repo_ids_to_try:
                            try:
                                print(f"   Trying GraphCast model from HuggingFace: {repo_id}")
                                
                                # List available files in the repo
                                try:
                                    files = list_repo_files(repo_id=repo_id, repo_type="model")
                                    
                                    # Find the .npz checkpoint file
                                    checkpoint_file = None
                                    for file in files:
                                        if file.endswith(".npz") and "GraphCast" in file:
                                            checkpoint_file = file
                                            break
                                except Exception as list_err:
                                    print(f"   ⚠️  Could not list files: {list_err}")
                                    # Try common filename patterns
                                    checkpoint_file = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
                                
                                if not checkpoint_file:
                                    # Try common filename patterns
                                    checkpoint_file = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
                                
                                print(f"   Downloading checkpoint: {checkpoint_file}")
                                checkpoint_path = hf_hub_download(
                                    repo_id=repo_id,
                                    filename=checkpoint_file,
                                    cache_dir="/data/models"
                                )
                                
                                # Load the checkpoint
                                print("   Loading GraphCast checkpoint...")
                                with open(checkpoint_path, "rb") as f:
                                    ckpt = checkpoint.load(f, gc_module.CheckPoint)
                                
                                # Store checkpoint data
                                self.checkpoint = ckpt
                                self.params = ckpt.params
                                self.model_config = ckpt.model_config
                                self.task_config = ckpt.task_config
                                
                                # Create the GraphCast model
                                self.model = gc_module.GraphCast(
                                    model_config=self.model_config,
                                    task_config=self.task_config
                                )
                                
                                # For GPU inference, we may need to override attention config
                                # (splash_attention is TPU-only, GPU needs triblockdiag_mha)
                                # This is handled automatically by the model, but we note it here
                                print("   ⚠️  Note: GPU inference uses triblockdiag_mha attention (TPU uses splash_attention)")
                                print("   ⚠️  GPU inference is slower and requires more memory than TPU")
                                
                                self.is_graphcast = True
                                model_loaded = True
                                print(f"   ✅ Loaded GraphCast from HuggingFace: {repo_id}")
                                print(f"   Model config: resolution={getattr(self.model_config, 'resolution', 'N/A')}")
                                
                                # Warn about resource requirements
                                if hasattr(self.model_config, 'resolution'):
                                    res = str(self.model_config.resolution)
                                    if '0.25' in res or '0p25' in res:
                                        print("   ⚠️  WARNING: 0.25deg model requires ~300GB RAM and ~60GB GPU vRAM")
                                    elif '1.0' in res or '1p0' in res or '1deg' in res:
                                        print("   ⚠️  WARNING: 1deg model requires ~24GB RAM and ~16GB GPU vRAM")
                                
                                break
                                
                            except Exception as e:
                                error_msg = str(e)
                                print(f"   ⚠️  Failed with {repo_id}: {error_msg}")
                                import traceback
                                print("   Full traceback:")
                                traceback.print_exc()
                                continue
                        
                        if not model_loaded:
                            print("   ⚠️  Could not load from any GraphCast repository")
                                
                    except ImportError as import_err:
                        print(f"   ⚠️  GraphCast/JAX libraries not available: {import_err}")
                        print("   These dependencies are installed via runtime_env but may take time on first deployment")
                        print("   Required packages:")
                        print("     - jax[cuda12] or jax (for CPU)")
                        print("     - jaxlib")
                        print("     - dm-haiku")
                        print("     - dm-tree")
                        print("     - git+https://github.com/google-deepmind/graphcast.git")
                        import traceback
                        traceback.print_exc()
                    except Exception as gc_err:
                        error_msg = str(gc_err)
                        print(f"   ⚠️  GraphCast loading failed: {error_msg}")
                        import traceback
                        traceback.print_exc()
                
                # 3. Try HuggingFace (as fallback or if model_path is a HF ID)
                if not model_loaded:
                    print("   Trying HuggingFace...")
                    try:
                        from transformers import AutoModel, AutoConfig
                        
                        # Try model_path as HuggingFace ID, or common alternatives
                        model_ids_to_try = [
                            model_path,  # User provided
                            "shermansiu/dm_graphcast",  # Known GraphCast model on HF
                            "shermansiu/dm_graphcast_small",  # Smaller version
                        ]
                        
                        for model_id in model_ids_to_try:
                            try:
                                print(f"   Trying HuggingFace model ID: {model_id}")
                                self.model = AutoModel.from_pretrained(
                                    model_id,
                                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                    device_map="auto",
                                    trust_remote_code=True
                                )
                                model_loaded = True
                                print(f"   ✅ Loaded from HuggingFace: {model_id}")
                                break
                            except Exception as e:
                                error_msg = str(e)
                                print(f"   ⚠️  Failed with {model_id}: {error_msg[:150]}")
                                continue
                    except Exception as hf_err:
                        print(f"   ⚠️  HuggingFace loading failed: {hf_err}")
                
                if not model_loaded:
                    # Provide helpful error message with alternatives
                    error_msg = (
                        f"Could not load weather model from: {model_path}\n"
                        f"\nTried:\n"
                        f"  1. Local directory: {model_path if model_path else 'N/A'}\n"
                        f"  2. Google's GraphCast library (requires JAX setup)\n"
                        f"  3. HuggingFace model IDs\n"
                        f"\nNote: 'google-deepmind/graphcast' is not a HuggingFace model.\n"
                        f"GraphCast from Google DeepMind uses JAX and requires special setup.\n"
                        f"\nAlternatives:\n"
                        f"  - Use a local GraphCast model directory\n"
                        f"  - Try: 'shermansiu/dm_graphcast' (if available on HuggingFace)\n"
                        f"  - Use a different weather model (FourCastNet, Pangu-Weather, etc.)\n"
                        f"  - Set up GraphCast manually from: https://github.com/google-deepmind/graphcast"
                    )
                    raise RuntimeError(error_msg)
                    
                # Verify GPU usage
                if torch.cuda.is_available():
                    print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
                    print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                else:
                    print("   ⚠️  CUDA not available - model may be on CPU")
                
                self.model_loaded = True
                self.loading_error = None
                
                # Discover schema after model is loaded
                self.schema = self._discover_model_schema()
                print(f"✅ Model loaded: {model_name}")
                print(f"   Discovered {len(self.schema.get('inputs', {}).get('required', []))} required inputs")
                print(f"   Discovered {len(self.schema.get('inputs', {}).get('optional', []))} optional inputs")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed to load model: {error_msg}")
                import traceback
                traceback.print_exc()
                self.model_loaded = False
                self.loading_error = error_msg
        
        def _discover_model_schema(self) -> Dict[str, Any]:
            """Dynamically discover the model's input/output schema"""
            schema = {
                "model_id": self.model_name,
                "model_class": type(self.model).__name__ if self.model else "Unknown",
                "capabilities": ["weather-forecasting"],
                "inputs": {
                    "required": [],
                    "optional": []
                },
                "methods": {}
            }
            
            # Try to inspect the forward method or __call__ method
            if self.model:
                try:
                    # Check if model has a forward method
                    if hasattr(self.model, 'forward'):
                        sig = inspect.signature(self.model.forward)
                        schema["methods"]["forward"] = self._parse_signature(sig)
                        
                        # Separate required vs optional
                        for param_name, param_info in schema["methods"]["forward"]["parameters"].items():
                            if param_name == "self":
                                continue
                            if param_info["required"]:
                                schema["inputs"]["required"].append(param_name)
                            else:
                                schema["inputs"]["optional"].append(param_name)
                except Exception as e:
                    schema["error"] = f"Could not inspect forward method: {e}"
            
            # Add common weather forecasting parameters
            if "latitude" not in schema["inputs"]["required"]:
                schema["inputs"]["optional"].append("latitude")
            if "longitude" not in schema["inputs"]["required"]:
                schema["inputs"]["optional"].append("longitude")
            if "forecast_hours" not in schema["inputs"]["optional"]:
                schema["inputs"]["optional"].append("forecast_hours")
            
            # Get default values
            schema["defaults"] = {
                "forecast_hours": 6,
                "latitude": 0.0,
                "longitude": 0.0,
            }
            
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
        
        def _get_resource_metrics(self) -> Dict[str, float]:
            """Get current resource usage metrics"""
            metrics = {}
            
            try:
                # CPU metrics
                process = psutil.Process()
                metrics["cpu_percent"] = process.cpu_percent(interval=0.1)
                metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
                metrics["num_cpus"] = psutil.cpu_count()
                
                # GPU metrics
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        
                        # Memory info
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics["gpu_memory_total_mb"] = mem_info.total / 1024 / 1024
                        metrics["gpu_memory_used_mb"] = mem_info.used / 1024 / 1024
                        metrics["gpu_memory_free_mb"] = mem_info.free / 1024 / 1024
                        
                        # Utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["gpu_utilization_percent"] = util.gpu
                        
                        # PyTorch allocated memory
                        metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
                        metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
                    except Exception:
                        # Fallback to PyTorch only
                        if torch.cuda.is_available():
                            metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
                            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
            except Exception:
                pass
            
            return metrics
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            if path.endswith("/forecast") or path.endswith("/predict") or path.endswith("/v1/forecast") or path.endswith("/v1/predict"):
                return await self.handle_forecast(request)
            elif path.endswith("/schema") or path.endswith("/v1/schema"):
                return await self.handle_schema(request)
            elif path.endswith("/models") or path.endswith("/v1/models"):
                return await self.handle_models(request)
            elif path.endswith("/health") or path.endswith("/v1/health"):
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
                
                if hasattr(self, 'loading_error') and self.loading_error:
                    response["loading_error"] = self.loading_error
                    response["error"] = f"Model loading failed: {self.loading_error}"
                
                return JSONResponse(response, status_code=503)
            
            if self.schema is None:
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
                    "owned_by": "graphcast"
                }]
            })
        
        async def handle_forecast(self, request: Request) -> Any:
            """Handle weather forecasting requests"""
            try:
                # Check if model is ready
                model_ready = (
                    hasattr(self, 'model_loaded') and 
                    self.model_loaded and 
                    hasattr(self, 'model') and 
                    self.model is not None
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
                
                # Build model inputs from request
                # GraphCast typically needs atmospheric data, coordinates, etc.
                model_inputs = self._build_model_inputs(data)
                
                # Generate forecast (different for GraphCast vs PyTorch models)
                if self.is_graphcast:
                    # GraphCast uses JAX - different inference pattern
                    import jax
                    # GraphCast inference requires specific input format
                    # This is a template - actual implementation needs proper data preprocessing
                    print("   ⚠️  GraphCast inference requires proper data preprocessing")
                    print("   ⚠️  See GraphCast demo notebook for proper input format")
                    # For now, return a placeholder
                    result = {
                        "forecast": "GraphCast inference requires ERA5/HRES data preprocessing",
                        "note": "See https://github.com/google-deepmind/graphcast for proper usage"
                    }
                else:
                    # PyTorch model inference
                    with torch.inference_mode():
                        result = self.model(**model_inputs)
                
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
                    "forecast": result,  # Actual format depends on GraphCast output
                    "parameters": data,
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
        def _build_model_inputs(self, data: Dict) -> Dict:
            """Build model inputs from request data"""
            # This is a template - actual GraphCast models may need specific input format
            inputs = {}
            
            # Common weather forecasting parameters
            if "latitude" in data:
                inputs["latitude"] = data["latitude"]
            if "longitude" in data:
                inputs["longitude"] = data["longitude"]
            if "forecast_hours" in data:
                inputs["forecast_hours"] = data["forecast_hours"]
            
            # Add other parameters from schema
            if self.schema and "inputs" in self.schema:
                all_inputs = self.schema["inputs"].get("required", []) + self.schema["inputs"].get("optional", [])
                for param_name in all_inputs:
                    if param_name in data:
                        inputs[param_name] = data[param_name]
            
            return inputs
    
    return GraphCastEndpoint


def create_app(model_name: str, model_path: str):
    """Create and return the Ray Serve application"""
    EndpointClass = create_deployment(model_name, model_path)
    return EndpointClass.bind()
