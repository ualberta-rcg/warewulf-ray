"""
Ray Serve deployment for Object Detection models (YOLO, etc.)
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

# Try to import torch separately
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not available. Install with: pip install pillow")


def create_deployment(model_name: str, model_path: str):
    """Create a deployment with unique name and autoscaling for a specific object detection model"""
    
    # Create unique deployment name from model name
    deployment_name = f"object-detection-{model_name.replace('/', '-').replace('_', '-').replace('.', '-')}"
    
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
                    "torchvision>=0.15.0",
                    "pillow>=9.0.0",
                    "fastapi>=0.100.0",
                    "psutil>=5.9.0",
                    "nvidia-ml-py>=12.0.0",
                    "ultralytics>=8.0.0",  # For YOLO models
                    "opencv-python>=4.8.0",  # For image processing
                    "numpy>=1.24.0",
                ],
                "env_vars": {
                    "HF_HOME": "/data/models",
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                }
            }
        }
    )
    class ObjectDetectionEndpoint:
        """Object detection endpoint with autoscaling and dynamic input discovery"""
        
        def __init__(self, model_name: str = model_name, model_path: str = model_path):
            import sys
            import subprocess
            
            self.model_loaded = False
            self.model = None
            self.model_name = model_name
            self.model_path = model_path
            self.schema = None  # Will be populated after model loads
            self.loading_error = None  # Store loading errors for API responses
            self.loading_started = False  # Track if loading has started
            self.is_yolo = False  # Track if it's a YOLO model
            
            # Start loading in background thread (non-blocking)
            loading_thread = threading.Thread(target=self._load_model, daemon=True)
            loading_thread.start()
        
        def _load_model(self):
            """Load the object detection model in background"""
            try:
                # Import torch in the function scope to avoid UnboundLocalError
                import torch
                
                self.loading_started = True
                print(f"🚀 Loading object detection model in background: {model_name}")
                print(f"   Model path: {model_path}")
                
                # Try to load as YOLO first (most common)
                try:
                    from ultralytics import YOLO
                    print("   Attempting to load as YOLO model...")
                    self.model = YOLO(model_path)
                    self.is_yolo = True
                    print("   ✅ Loaded as YOLO model")
                except Exception as yolo_err:
                    print(f"   ⚠️  Not a YOLO model: {yolo_err}")
                    # Try transformers object detection
                    try:
                        from transformers import AutoModelForObjectDetection, AutoImageProcessor
                        print("   Attempting to load as transformers object detection model...")
                        self.model = AutoModelForObjectDetection.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto"
                        )
                        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
                        self.is_yolo = False
                        print("   ✅ Loaded as transformers object detection model")
                    except Exception as tf_err:
                        raise RuntimeError(f"Failed to load as YOLO or transformers: YOLO error: {yolo_err}, Transformers error: {tf_err}")
                
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
                "model_type": "yolo" if self.is_yolo else "transformers",
                "capabilities": ["object-detection"],
                "inputs": {
                    "required": ["image"],
                    "optional": ["confidence", "iou_threshold", "image_size"]
                },
                "methods": {}
            }
            
            # Add YOLO-specific parameters if applicable
            if self.is_yolo:
                schema["inputs"]["optional"].extend([
                    "conf",  # Confidence threshold
                    "iou",  # IoU threshold
                    "imgsz",  # Image size
                    "max_det",  # Max detections
                    "agnostic_nms",  # Class-agnostic NMS
                ])
                schema["defaults"] = {
                    "confidence": 0.25,
                    "iou_threshold": 0.45,
                    "image_size": 640,
                }
            else:
                schema["defaults"] = {
                    "confidence": 0.5,
                    "iou_threshold": 0.5,
                    "image_size": None,
                }
            
            return schema
        
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
        
        def _decode_base64_image(self, image_data: str):
            """Decode base64 image data"""
            from PIL import Image
            
            # Handle data URI format: "data:image/png;base64,..."
            if image_data.startswith("data:"):
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            if path.endswith("/detect") or path.endswith("/predict") or path.endswith("/v1/detect") or path.endswith("/v1/predict"):
                return await self.handle_detect(request)
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
                    "owned_by": "object-detection"
                }]
            })
        
        async def handle_detect(self, request: Request) -> Any:
            """Handle object detection requests"""
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
                
                # Decode image
                if "image" not in data:
                    return JSONResponse({"error": "Missing 'image' parameter (base64 encoded)"}, status_code=400)
                
                image = self._decode_base64_image(data["image"])
                
                # Get parameters
                confidence = data.get("confidence", self.schema.get("defaults", {}).get("confidence", 0.25) if self.is_yolo else 0.5)
                iou_threshold = data.get("iou_threshold", self.schema.get("defaults", {}).get("iou_threshold", 0.45) if self.is_yolo else 0.5)
                
                # Run detection
                with torch.inference_mode():
                    if self.is_yolo:
                        # YOLO model
                        results = self.model(
                            image,
                            conf=confidence,
                            iou=iou_threshold,
                            imgsz=data.get("image_size", 640),
                            max_det=data.get("max_det", 300),
                            agnostic_nms=data.get("agnostic_nms", False)
                        )
                        
                        # Format YOLO results
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            for i in range(len(boxes)):
                                detections.append({
                                    "class": int(boxes.cls[i]),
                                    "class_name": result.names[int(boxes.cls[i])],
                                    "confidence": float(boxes.conf[i]),
                                    "bbox": boxes.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                                })
                    else:
                        # Transformers model
                        inputs = self.image_processor(image, return_tensors="pt")
                        if torch.cuda.is_available():
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        
                        # Process outputs (simplified - actual format depends on model)
                        target_sizes = torch.tensor([image.size[::-1]])
                        results = self.image_processor.post_process_object_detection(
                            outputs, threshold=confidence, target_sizes=target_sizes
                        )[0]
                        
                        detections = []
                        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                            if score >= confidence:
                                detections.append({
                                    "class": int(label),
                                    "class_name": self.model.config.id2label[int(label)],
                                    "confidence": float(score),
                                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                                })
                
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
                    "detections": detections,
                    "detection_count": len(detections),
                    "parameters": {
                        "confidence": confidence,
                        "iou_threshold": iou_threshold,
                    },
                    "compute_metrics": compute_metrics
                })
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )


def create_app(model_name: str, model_path: str):
    """Create and return the Ray Serve application"""
    EndpointClass = create_deployment(model_name, model_path)
    return EndpointClass.bind()
