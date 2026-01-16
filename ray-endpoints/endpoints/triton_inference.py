#!/usr/bin/env python3
"""
Deploy Triton Inference Server on Ray cluster with:
- API key authentication
- Resource usage logging
- Auto-scaling
- Multi-model support
"""

import time
import logging
import subprocess
import psutil
import json
import ray
from ray import serve
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import numpy as np

# Initialize Ray
#ray.init(address="auto")

# Setup logging
LOG_DIR = "/data/logs"
subprocess.run(["mkdir", "-p", LOG_DIR], check=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/triton_usage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Keys - Generate secure keys for production!
API_KEYS = {
    "demo-key-admin": "admin_user",
    "demo-key-user1": "demo_user_1",
    "demo-key-user2": "demo_user_2",
}

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[credentials.credentials]

app = FastAPI(
    title="Ray-Triton Inference API",
    description="Multi-model inference with authentication and usage tracking",
    version="1.0.0"
)

@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 10,
    },
    health_check_period_s=10,
    health_check_timeout_s=30
)
@serve.ingress(app)
class TritonInferenceDeployment:
    def __init__(self, model_repo_path: str = "/data/ray-triton/model_repository"):
        self.model_repo_path = model_repo_path
        self.process = None
        self.triton_http_port = 8001
        self.triton_grpc_port = 8002

        logger.info("Starting Triton Inference Server deployment...")

        # Start warmup timer
        warmup_start = time.time()

        # Start Triton server
        self.start_triton_server()

        # Wait for server ready
        self.wait_for_server_ready(timeout=300)

        warmup_time = time.time() - warmup_start

        # Log initial resources and warmup
        self.log_resources(
            user="system",
            model="deployment_init",
            inference_time=0,
            warmup_time=warmup_time
        )

        logger.info(f"Deployment warmup completed in {warmup_time:.2f}s")

        # Load model catalog
        self.load_model_catalog()

    def start_triton_server(self):
        """Start Triton server as subprocess"""
        self.process = subprocess.Popen(
            [
                "/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver",
                "--model-repository", self.model_repo_path,
                "--http-port", str(self.triton_http_port),
                "--grpc-port", str(self.triton_grpc_port),
                "--metrics-port", "8003",
                "--http-address", "0.0.0.0",
                "--log-verbose", "1",
                "--model-control-mode", "poll",
                "--repository-poll-secs", "30"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Triton server process started (PID: {self.process.pid})")

    def wait_for_server_ready(self, timeout: int = 300):
        """Poll Triton server until ready"""
        import requests
        url = f"http://localhost:{self.triton_http_port}/v2/health/ready"
        start = time.time()

        while time.time() - start < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    logger.info("Triton server is ready!")
                    return
            except:
                pass
            time.sleep(2)

        raise TimeoutError("Triton server failed to become ready")

    def load_model_catalog(self):
        """Load model catalog if available, or auto-discover ONNX models"""
        catalog_path = f"{self.model_repo_path}/model_catalog.json"
        try:
            with open(catalog_path, 'r') as f:
                self.model_catalog = json.load(f)
            logger.info(f"Loaded model catalog with {len(self.model_catalog.get('models', []))} models")
        except:
            # Auto-discover models in repository
            self.model_catalog = self.discover_models()
            if self.model_catalog:
                logger.info(f"Auto-discovered {len(self.model_catalog.get('models', []))} models in repository")
            else:
                logger.warning("No model catalog found and no models discovered")
    
    def discover_models(self):
        """Auto-discover ONNX and other models in the repository"""
        import os
        from pathlib import Path
        
        models = []
        repo_path = Path(self.model_repo_path)
        
        if not repo_path.exists():
            logger.warning(f"Model repository not found: {self.model_repo_path}")
            return None
        
        # Scan repository for model directories
        for model_dir in repo_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            config_file = model_dir / "config.pbtxt"
            version_dir = model_dir / "1"
            
            # Check if it's a valid Triton model directory
            if not config_file.exists():
                continue
            
            # Determine model type by checking for model files
            model_type = "unknown"
            model_files = []
            
            if version_dir.exists():
                for model_file in version_dir.iterdir():
                    if model_file.is_file():
                        model_files.append(model_file.name)
                        ext = model_file.suffix.lower()
                        if ext == ".onnx":
                            model_type = "onnx"
                        elif ext in [".plan", ".engine"]:
                            model_type = "tensorrt"
                        elif ext == ".pt" or ext == ".pth":
                            model_type = "pytorch"
                        elif ext == ".pb":
                            model_type = "tensorflow"
            
            models.append({
                "name": model_name,
                "type": model_type,
                "path": str(model_dir),
                "files": model_files,
                "has_config": config_file.exists()
            })
        
        if models:
            return {
                "models": models,
                "repository_path": str(repo_path),
                "discovery_method": "auto"
            }
        
        return None

    def get_gpu_info(self):
        """Get GPU information"""
        try:
            output = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]).decode().strip()

            gpus = []
            for line in output.split('\n'):
                parts = line.split(', ')
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'memory_total_mb': parts[2],
                    'memory_used_mb': parts[3],
                    'utilization_pct': parts[4]
                })
            return gpus
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return []

    def log_resources(self, user: str, model: str, inference_time: float, warmup_time: float = 0):
        """Log resource usage and metrics"""
        gpus = self.get_gpu_info()
        cpu_cores = psutil.cpu_count()
        mem = psutil.virtual_memory()

        log_entry = {
            "timestamp": time.time(),
            "user": user,
            "model": model,
            "inference_time_sec": round(inference_time, 4),
            "warmup_time_sec": round(warmup_time, 4),
            "num_gpus": len(gpus),
            "gpus": gpus,
            "cpu_cores": cpu_cores,
            "system_memory_total_gb": round(mem.total / (1024**3), 2),
            "system_memory_used_gb": round(mem.used / (1024**3), 2),
            "system_memory_percent": mem.percent
        }

        logger.info(f"USAGE: {json.dumps(log_entry)}")

    @app.get("/")
    async def root(self, user: str = Depends(verify_api_key)):
        """API root with model listing"""
        return {
            "service": "Ray-Triton Inference API",
            "user": user,
            "model_catalog": self.model_catalog,
            "endpoints": {
                "health": "/health",
                "models": "/v2/models",
                "infer": "/v2/models/{model_name}/infer"
            }
        }

    @app.get("/health")
    async def health(self):
        """Health check (no auth required)"""
        import requests
        try:
            response = requests.get(f"http://localhost:{self.triton_http_port}/v2/health/ready")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "triton_ready": response.status_code == 200
            }
        except:
            return {"status": "unhealthy", "triton_ready": False}

    @app.get("/v2/models")
    async def list_models(self, user: str = Depends(verify_api_key)):
        """List all available models from Triton Server"""
        import requests
        try:
            # Get models from Triton Server
            response = requests.get(f"http://localhost:{self.triton_http_port}/v2/models")
            if response.status_code == 200:
                triton_models = response.json()
            else:
                triton_models = []
            
            # Combine with discovered catalog
            return {
                "user": user,
                "triton_models": triton_models,
                "discovered_catalog": self.model_catalog,
                "repository_path": self.model_repo_path,
                "total_models": len(triton_models) if isinstance(triton_models, list) else 0
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            # Return discovered catalog even if Triton is not responding
            return {
                "user": user,
                "triton_models": [],
                "discovered_catalog": self.model_catalog,
                "repository_path": self.model_repo_path,
                "error": str(e),
                "note": "Triton server may not be ready yet"
            }

    @app.post("/v2/models/{model_name}/infer")
    async def infer(
        self,
        model_name: str,
        request: Request,
        user: str = Depends(verify_api_key)
    ):
        """Run inference on specified model"""
        import requests

        start_time = time.time()

        try:
            # Get request body
            body = await request.json()

            # Forward to Triton
            triton_url = f"http://localhost:{self.triton_http_port}/v2/models/{model_name}/infer"
            response = requests.post(triton_url, json=body, timeout=30)

            inference_time = time.time() - start_time

            # Log usage
            self.log_resources(user, model_name, inference_time)

            if response.status_code == 200:
                result = response.json()
                result['inference_time'] = inference_time
                result['user'] = user
                return result
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Triton error: {response.text}"
                )

        except Exception as e:
            inference_time = time.time() - start_time
            self.log_resources(user, model_name, inference_time)
            raise HTTPException(status_code=500, detail=str(e))

    async def __del__(self):
        """Cleanup on shutdown"""
        if self.process:
            logger.info("Shutting down Triton server...")
            self.process.terminate()
            self.process.wait(timeout=10)

# Deploy
MODEL_REPO = "/data/ray-triton/model_repository"

deployment = TritonInferenceDeployment.bind(MODEL_REPO)
handle = serve.run(deployment, name="triton-inference", route_prefix="/")

print("\n" + "="*70)
print("RAY-TRITON DEPLOYMENT READY!")
print("="*70)
print(f"\nModel Repository: {MODEL_REPO}")
print(f"Logs: {LOG_DIR}/triton_usage.log")
print(f"\nAPI Keys for demo:")
for key, user in API_KEYS.items():
    print(f"  {user}: {key}")
print(f"\nEndpoints:")
print(f"  Health: curl http://localhost:8000/health")
print(f"  List models: curl -H 'Authorization: Bearer demo-key-admin' http://localhost:8000/v2/models")
print(f"  Inference: POST http://localhost:8000/v2/models/{{model_name}}/infer")
print(f"\nRay Dashboard: http://<head-ip>:8265")
print(f"Triton Metrics: http://localhost:8003/metrics")
print("="*70 + "\n")
