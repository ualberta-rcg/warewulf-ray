"""
Stable Diffusion endpoint using Triton Server (based on Ray Serve Triton integration guide)
Deploy individually: /opt/ray/bin/python stable_diffusion_triton.py
"""
import numpy
from pathlib import Path
from fastapi import FastAPI
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠ PIL/Pillow not available - install with: /opt/ray/bin/pip install Pillow")
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

# Try to import diffusers for direct model loading
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠ diffusers not available - install with: /opt/ray/bin/pip install diffusers torch")
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
        print("Loading Stable Diffusion...")
        
        self._triton_server = None
        self._stable_diffusion = None
        self._pipeline = None
        self._use_triton = False
        self._model_path = None
        
        # Model directory paths
        model_dir = Path("/data/models/stablediffusion")
        triton_repo = Path("/data/ray-triton/model_repository")
        
        # Check if we have a Triton model repository structure
        # First check the dedicated Triton repository, then check model_dir
        has_triton_structure = (
            (triton_repo / "stable_diffusion").exists() or
            (triton_repo / "text_encoder").exists() or
            (triton_repo / "vae").exists() or
            (model_dir / "stable_diffusion").exists() or
            (model_dir / "text_encoder").exists() or
            (model_dir / "vae").exists()
        )
        
        # Determine which repository to use
        triton_repo_path = str(triton_repo) if triton_repo.exists() else str(model_dir)
        
        # Try Triton if available and structure exists
        if TRITON_AVAILABLE and has_triton_structure:
            try:
                print(f"📦 Using Triton Server (repository: {triton_repo_path})")
                self._triton_server = tritonserver.Server(
                    model_repository=[triton_repo_path],
                    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
                    log_info=False,
                )
                self._triton_server.start(wait_until_ready=True)
                self._use_triton = True
                self._ready = True
                print("✓ Triton Server started successfully")
                return
            except Exception as e:
                print(f"⚠ Failed to start Triton Server: {e}")
                print("  Falling back to direct model loading...")
        
        # Fall back to direct model loading with diffusers
        if DIFFUSERS_AVAILABLE:
            try:
                print("📦 Using diffusers (direct model loading)")
                # Find available SD models
                model_files = list(model_dir.glob("*.safetensors"))
                model_files.extend(list(model_dir.glob("*.ckpt")))
                
                if not model_files:
                    print("⚠ No model files found")
                    self._ready = False
                    return
                
                # Prefer v1-5 or sd_xl models
                preferred = [m for m in model_files if "v1-5" in m.name or "sd_xl" in m.name]
                if preferred:
                    model_file = preferred[0]
                else:
                    model_file = model_files[0]
                
                print(f"📦 Loading model: {model_file.name}")
                self._model_path = str(model_file)
                
                # Load pipeline from single file
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._pipeline = StableDiffusionPipeline.from_single_file(
                    str(model_file),
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self._pipeline = self._pipeline.to(device)
                self._use_triton = False
                self._ready = True
                print("✓ Model loaded successfully with diffusers")
                return
            except Exception as e:
                print(f"✗ Failed to load model with diffusers: {e}")
                import traceback
                traceback.print_exc()
        
        # If we get here, nothing worked
        print("✗ No working model loading method available")
        self._ready = False
    
    @app.get("/health")
    def health(self):
        """Health check endpoint"""
        return {
            "status": "ready" if self._ready else "loading",
            "model": "stable-diffusion-triton",
            "backend": "triton" if self._use_triton else "diffusers" if self._pipeline else "none",
            "triton_available": TRITON_AVAILABLE and self._triton_server is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE and self._pipeline is not None
        }
    
    @app.get("/generate")
    def generate(self, prompt: str, filename: str = None, steps: int = 20, guidance_scale: float = 7.5) -> Dict[str, Any]:
        """
        Generate image from prompt using Stable Diffusion
        
        Args:
            prompt: Text prompt for image generation
            filename: Optional filename to save image (default: auto-generated)
            steps: Number of inference steps (default: 20)
            guidance_scale: Guidance scale (default: 7.5)
        
        Returns:
            JSON response with image data and usage stats
        """
        start_time = time.time()
        
        if not self._ready:
            return {
                "error": "Model not ready",
                "status": "error"
            }
        
        try:
            generated_image = None
            
            if self._use_triton and self._triton_server:
                # Use Triton Server
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
                
                # Generate via Triton
                for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
                    generated_image = (
                        numpy.from_dlpack(response.outputs["generated_image"])
                        .squeeze()
                        .astype(numpy.uint8)
                    )
            elif self._pipeline:
                # Use diffusers directly
                print(f"Generating image with prompt: {prompt[:50]}...")
                result = self._pipeline(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale
                )
                generated_image = numpy.array(result.images[0])
            else:
                return {
                    "error": "No model backend available",
                    "status": "error"
                }
            
            if generated_image is None:
                return {
                    "error": "No image generated",
                    "status": "error"
                }
            
            # Convert to PIL Image (if available) or save as numpy array
            if PIL_AVAILABLE:
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
            else:
                # Fallback: save as numpy array
                if filename:
                    saved_path = filename
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = f"generated_image_{timestamp}.npy"
                numpy.save(saved_path, generated_image)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Save metrics
            self._save_metrics(prompt, latency_ms, "success")
            
            return {
                "status": "success",
                "prompt": prompt,
                "image_path": saved_path,
                "backend": "triton" if self._use_triton else "diffusers",
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
            "backend": "triton" if self._use_triton else "diffusers" if self._pipeline else "none",
            "model_path": self._model_path,
            "triton_available": TRITON_AVAILABLE and self._triton_server is not None,
            "diffusers_available": DIFFUSERS_AVAILABLE and self._pipeline is not None
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
