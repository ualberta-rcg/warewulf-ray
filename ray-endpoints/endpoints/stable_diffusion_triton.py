"""
Stable Diffusion endpoint using Triton Server (based on Ray Serve Triton integration guide)
Deploy individually: python stable_diffusion_triton.py
"""
import numpy
from pathlib import Path
from fastapi import FastAPI
from PIL import Image
from ray import serve
from ray.serve.config import AutoscalingConfig
import time
import json
from datetime import datetime
from typing import Dict, Any

# Try to import tritonserver - will fail gracefully if not available
try:
    import tritonserver
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠ tritonserver not available - using placeholder mode")

app = FastAPI()

@serve.deployment(
    name="stable-diffusion-triton",
    ray_actor_options={"num_gpus": 1},
    autoscaling_config=AutoscalingConfig(
        min_replicas=0,
        max_replicas=5,
        target_num_ongoing_requests_per_replica=1,
        downscale_delay_s=120,
        upscale_delay_s=0,
    ),
)
@serve.ingress(app)
class StableDiffusionTriton:
    def __init__(self):
        print("Loading Stable Diffusion with Triton Server...")
        
        if not TRITON_AVAILABLE:
            print("⚠ Triton Server not available - using placeholder")
            self._triton_server = None
            self._ready = False
            return
        
        # Model repository path - adjust to your actual path
        # Can be local directory or S3 path (e.g., "s3://bucket/models")
        model_repository = "/data/models/Stable-diffusion"
        
        # Check if we have a Triton model repository structure
        # If not, we'll use the safetensors files directly
        if not Path(model_repository).exists():
            print(f"⚠ Model repository not found at {model_repository}")
            print("  Using placeholder mode - configure model_repository path")
            self._triton_server = None
            self._ready = False
            return
        
        try:
            self._triton_server = tritonserver.Server(
                model_repository=[model_repository],
                model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
                log_info=False,
            )
            self._triton_server.start(wait_until_ready=True)
            self._stable_diffusion = None
            self._ready = True
            print("✓ Triton Server started successfully")
        except Exception as e:
            print(f"✗ Failed to start Triton Server: {e}")
            self._triton_server = None
            self._ready = False
    
    @app.get("/health")
    def health(self):
        """Health check endpoint"""
        return {
            "status": "ready" if self._ready else "loading",
            "model": "stable-diffusion-triton",
            "triton_available": TRITON_AVAILABLE and self._triton_server is not None
        }
    
    @app.get("/generate")
    def generate(self, prompt: str, filename: str = None) -> Dict[str, Any]:
        """
        Generate image from prompt using Stable Diffusion via Triton Server
        
        Args:
            prompt: Text prompt for image generation
            filename: Optional filename to save image (default: auto-generated)
        
        Returns:
            JSON response with image data and usage stats
        """
        start_time = time.time()
        
        if not self._ready or self._triton_server is None:
            return {
                "error": "Triton Server not available",
                "status": "error"
            }
        
        # Load models if not already loaded
        if not self._triton_server.model("stable_diffusion").ready():
            try:
                print("Loading models into Triton...")
                self._triton_server.load("text_encoder")
                self._triton_server.load("vae")
                self._stable_diffusion = self._triton_server.load("stable_diffusion")
                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready after loading")
            except Exception as error:
                print(f"Error loading stable diffusion model: {error}")
                return {
                    "error": f"Failed to load model: {error}",
                    "status": "error"
                }
        
        try:
            # Generate image
            generated_image = None
            for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
                generated_image = (
                    numpy.from_dlpack(response.outputs["generated_image"])
                    .squeeze()
                    .astype(numpy.uint8)
                )
            
            if generated_image is None:
                return {
                    "error": "No image generated",
                    "status": "error"
                }
            
            # Convert to PIL Image
            image = Image.fromarray(generated_image)
            
            # Save if filename provided
            if filename:
                image.save(filename)
                saved_path = filename
            else:
                # Auto-generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = f"generated_image_{timestamp}.jpg"
                image.save(saved_path)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Save metrics
            self._save_metrics(prompt, latency_ms, "success")
            
            return {
                "status": "success",
                "prompt": prompt,
                "image_path": saved_path,
                "usage_stats": {
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self._save_metrics(prompt, latency_ms, "error")
            
            return {
                "error": str(e),
                "status": "error",
                "usage_stats": {
                    "latency_ms": round(latency_ms, 2)
                }
            }
    
    def _save_metrics(self, prompt: str, latency_ms: float, status: str):
        """Save usage metrics to file"""
        metrics_path = Path("/var/log/ray/sd-triton-metrics")
        metrics_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt[:100],  # Truncate
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "model": "stable-diffusion-triton"
        }
        
        metrics_file = metrics_path / f"metrics-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    @app.get("/info")
    def info(self):
        """Get deployment information"""
        return {
            "name": "stable-diffusion-triton",
            "route": "/",
            "ready": self._ready,
            "triton_available": TRITON_AVAILABLE and self._triton_server is not None
        }


if __name__ == "__main__":
    # Deploy the deployment
    print("🚀 Deploying Stable Diffusion with Triton Server...")
    serve.run(
        StableDiffusionTriton.bind(),
        name="stable-diffusion-triton",
        route_prefix="/api/v1/stable-diffusion-triton"
    )
    
    print("✅ Deployment complete!")
    print("📡 Endpoint available at: http://localhost:8000/api/v1/stable-diffusion-triton")
    print("🔍 Health check: http://localhost:8000/api/v1/stable-diffusion-triton/health")
    print("🎨 Generate: http://localhost:8000/api/v1/stable-diffusion-triton/generate?prompt=your+prompt")
