"""
Stable Diffusion endpoint with polling, queueing, and usage stats
"""
import sys
from pathlib import Path

# Handle imports - try relative first, then absolute
try:
    from ..base import BaseEndpoint
except ImportError:
    # If relative import fails, add parent to path and import
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from base import BaseEndpoint
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import time
import json
import os
from datetime import datetime
from pathlib import Path
import asyncio
from collections import deque
import threading

class StableDiffusionEndpoint(BaseEndpoint):
    ROUTE_PREFIX = "/api/v1/stable-diffusion"
    DEPLOYMENT_NAME = "stable-diffusion"
    MIN_REPLICAS = 0
    MAX_REPLICAS = 5
    DOWNSCALE_DELAY_S = 120  # Keep alive longer for SD models
    
    def __init__(self):
        # Model paths
        self.model_base_path = Path("/data/models/Stable-diffusion")
        self.metrics_path = Path("/var/log/ray/sd-metrics")
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        # Request queue
        self.request_queue = deque()
        self.queue_lock = threading.Lock()
        
        # Load model
        self.model = self.load_model()
        self.ready = True
        print(f"✓ {self.DEPLOYMENT_NAME} ready!")
    
    def load_model(self):
        """Load Stable Diffusion model"""
        print("Loading Stable Diffusion model...")
        
        # Find available SD models
        model_files = list(self.model_base_path.glob("*.safetensors"))
        model_files.extend(list(self.model_base_path.glob("*.ckpt")))
        
        if not model_files:
            print("⚠ No model files found, using placeholder")
            return {"model_type": "stable-diffusion", "loaded": False, "models_available": 0}
        
        # Prefer v1-5 or sd_xl models
        preferred = [m for m in model_files if "v1-5" in m.name or "sd_xl" in m.name]
        if preferred:
            model_file = preferred[0]
        else:
            model_file = model_files[0]
        
        print(f"📦 Selected model: {model_file.name}")
        
        # Simulate loading (replace with actual model loading)
        # For actual implementation, you'd use diffusers or similar:
        # from diffusers import StableDiffusionPipeline
        # self.pipeline = StableDiffusionPipeline.from_single_file(str(model_file))
        # self.pipeline = self.pipeline.to("cuda")
        
        time.sleep(5)  # Simulate loading time
        
        return {
            "model_type": "stable-diffusion",
            "model_file": str(model_file),
            "loaded": True,
            "models_available": len(model_files)
        }
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stable Diffusion inference"""
        prompt = data.get("prompt", "")
        steps = data.get("steps", 20)
        width = data.get("width", 512)
        height = data.get("height", 512)
        guidance_scale = data.get("guidance_scale", 7.5)
        
        # Actual inference would be:
        # image = self.pipeline(
        #     prompt=prompt,
        #     num_inference_steps=steps,
        #     width=width,
        #     height=height,
        #     guidance_scale=guidance_scale
        # ).images[0]
        # image_base64 = image_to_base64(image)
        
        return {
            "model": "stable-diffusion",
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "image": "base64_encoded_image_placeholder",
            "model_file": self.model.get("model_file", "unknown")
        }
    
    def save_metrics(self, request_data: Dict, response_data: Dict, 
                     start_time: float, end_time: float):
        """Save usage metrics to file for later collection"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_data.get("request_id", "unknown"),
            "prompt": request_data.get("prompt", "")[:100],  # Truncate
            "steps": request_data.get("steps", 20),
            "width": request_data.get("width", 512),
            "height": request_data.get("height", 512),
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "model": self.DEPLOYMENT_NAME,
            "response_size": len(str(response_data)),
            "status": "success" if "error" not in response_data else "error"
        }
        
        # Append to metrics file (one file per day)
        metrics_file = self.metrics_path / f"metrics-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app with queueing and metrics"""
        app = FastAPI(title=self.DEPLOYMENT_NAME)
        
        @app.get("/health")
        async def health():
            return {
                "status": "ready" if self.ready else "loading",
                "model": self.DEPLOYMENT_NAME,
                "queue_size": len(self.request_queue)
            }
        
        @app.post("/generate")
        async def generate(request: Request, background_tasks: BackgroundTasks):
            """Generate image with queueing and metrics"""
            start_time = time.time()
            request_data = await request.json()
            
            # Add request ID if not present
            if "request_id" not in request_data:
                request_data["request_id"] = f"req_{int(time.time() * 1000)}"
            
            # Poll until model is ready (if scaled to zero)
            max_wait = 300
            poll_interval = 1.0
            waited = 0
            
            while not self.ready and waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                if waited % 10 == 0:
                    print(f"⏳ Waiting for model to load... ({int(waited)}s)")
            
            if not self.ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Model failed to load",
                        "request_id": request_data.get("request_id")
                    }
                )
            
            # Add to queue
            with self.queue_lock:
                queue_position = len(self.request_queue)
                self.request_queue.append(request_data)
            
            try:
                # Process request
                result = self.predict(request_data)
                end_time = time.time()
                
                # Add usage stats to response
                result["usage_stats"] = {
                    "latency_ms": round((end_time - start_time) * 1000, 2),
                    "queue_wait_ms": 0,
                    "model_load_time_ms": waited * 1000 if waited > 0 else 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Save metrics in background
                background_tasks.add_task(
                    self.save_metrics,
                    request_data,
                    result,
                    start_time,
                    end_time
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                error_response = {
                    "error": str(e),
                    "request_id": request_data.get("request_id"),
                    "usage_stats": {
                        "latency_ms": round((end_time - start_time) * 1000, 2),
                        "status": "error"
                    }
                }
                
                background_tasks.add_task(
                    self.save_metrics,
                    request_data,
                    error_response,
                    start_time,
                    end_time
                )
                
                return JSONResponse(
                    status_code=500,
                    content=error_response
                )
            finally:
                # Remove from queue
                with self.queue_lock:
                    if self.request_queue and self.request_queue[0].get("request_id") == request_data.get("request_id"):
                        self.request_queue.popleft()
        
        @app.get("/metrics/summary")
        async def metrics_summary():
            """Get summary of collected metrics"""
            metrics_file = self.metrics_path / f"metrics-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
            
            if not metrics_file.exists():
                return {"message": "No metrics collected today"}
            
            metrics = []
            with open(metrics_file, "r") as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
            
            if not metrics:
                return {"message": "No metrics available"}
            
            latencies = [m["latency_ms"] for m in metrics if "latency_ms" in m]
            
            return {
                "total_requests": len(metrics),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "successful": sum(1 for m in metrics if m.get("status") == "success"),
                "errors": sum(1 for m in metrics if m.get("status") == "error")
            }
        
        @app.get("/info")
        async def info():
            return {
                "name": self.DEPLOYMENT_NAME,
                "route": self.ROUTE_PREFIX,
                "ready": self.ready,
                "queue_size": len(self.request_queue),
                "model_info": self.model
            }
        
        return app
