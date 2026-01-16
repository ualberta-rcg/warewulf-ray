"""
Triton Inference Server deployment with API key authentication and resource logging
Based on start-triton.py
"""
import time
import logging
import subprocess
import psutil
import json
import sys
from pathlib import Path

# Handle imports - ensure we can find base if needed
try:
    from ..base import BaseEndpoint
except ImportError:
    # If relative import fails, add parent to path and import
    _parent = Path(__file__).parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from base import BaseEndpoint

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import numpy as np
from ray import serve
from ray.serve.config import AutoscalingConfig

# Setup logging
LOG_DIR = Path("/data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_DIR / "triton_usage.log")),
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

# Setup runtime environment for dependencies
import os
from pathlib import Path

# Get ray-endpoints directory for PYTHONPATH
ray_endpoints_dir = Path(__file__).parent.parent.absolute()

# Check for shared venv on NFS (more efficient than installing on each worker)
shared_venv_path = "/data/ray-endpoints-venv"
use_shared_venv = os.path.exists(shared_venv_path) and os.path.exists(f"{shared_venv_path}/bin/python")

# Build PATH with shared venv if available
path_components = [os.environ.get("PATH", "")]
if use_shared_venv:
    path_components.insert(0, f"{shared_venv_path}/bin")

# Runtime environment - ensures dependencies are available on worker nodes
runtime_env = {
    "env_vars": {
        "PYTHONPATH": str(ray_endpoints_dir),
        "PATH": ":".join(path_components),
    },
    # Add pip packages if needed (uncomment and add)
    # Note: If using shared venv, install packages there instead:
    # /data/ray-endpoints-venv/bin/pip install package_name
    # "pip": ["psutil", "requests", "numpy"],  # Only use if NOT using shared venv
}

# Support Compute Canada modules
compute_canada_modules = []
if os.path.exists("/cvmfs/soft.computecanada.ca/config/profile/bash.sh"):
    # Compute Canada modules available - add modules here if needed
    # Example: compute_canada_modules = ["python/3.12", "cuda/12.0", "cudnn/8.9"]
    pass

@serve.deployment(
    name="triton-inference",
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=4,
        target_num_ongoing_requests_per_replica=10,
        downscale_delay_s=300,  # Keep alive longer
        upscale_delay_s=0,
    ),
    health_check_period_s=10,
    health_check_timeout_s=30,
    runtime_env=runtime_env,
)
@serve.ingress(app)
class TritonInferenceDeployment:
    def __init__(self, model_repo_path: str = "/data/ray-triton/model_repository"):
        # Ensure Python path is set (important for worker nodes)
        import sys
        if str(ray_endpoints_dir) not in sys.path:
            sys.path.insert(0, str(ray_endpoints_dir))
        
        # Activate shared venv if available (NFS share)
        if use_shared_venv:
            try:
                # Add venv to sys.path
                venv_site_packages = f"{shared_venv_path}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
                if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
                    sys.path.insert(0, venv_site_packages)
                logger.info(f"Using shared venv from NFS: {shared_venv_path}")
            except Exception as e:
                logger.warning(f"Could not use shared venv: {e}")
        
        # Load Compute Canada modules if specified
        if compute_canada_modules:
            try:
                import subprocess
                # Source the module system
                subprocess.run(
                    ["bash", "-c", "source /cvmfs/soft.computecanada.ca/config/profile/bash.sh && " + 
                     " && ".join([f"module load {m}" for m in compute_canada_modules])],
                    check=True
                )
                logger.info(f"Loaded Compute Canada modules: {compute_canada_modules}")
            except Exception as e:
                logger.warning(f"Could not load Compute Canada modules: {e}")
        
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
        triton_binary = "/opt/ray/lib/python3.12/site-packages/pytriton/tritonserver/bin/tritonserver"
        
        if not Path(triton_binary).exists():
            raise FileNotFoundError(f"Triton server binary not found at {triton_binary}")
        
        self.process = subprocess.Popen(
            [
                triton_binary,
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
        """Load model catalog if available"""
        catalog_path = Path(self.model_repo_path) / "model_catalog.json"
        try:
            with open(catalog_path, 'r') as f:
                self.model_catalog = json.load(f)
            logger.info(f"Loaded model catalog with {len(self.model_catalog['models'])} models")
        except:
            self.model_catalog = None
            logger.warning("No model catalog found")

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
        """List all available models"""
        import requests
        try:
            response = requests.get(f"http://localhost:{self.triton_http_port}/v2/models")
            models = response.json()
            return {
                "user": user,
                "models": models,
                "catalog": self.model_catalog
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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


if __name__ == "__main__":
    # Deploy the deployment
    print("🚀 Deploying Triton Inference Server...")
    
    MODEL_REPO = "/data/ray-triton/model_repository"
    
    deployment = TritonInferenceDeployment.bind(MODEL_REPO)
    serve.run(deployment, name="triton-inference", route_prefix="/api/v1/triton-inference")
    
    print("\n" + "="*70)
    print("RAY-TRITON DEPLOYMENT READY!")
    print("="*70)
    print(f"\nModel Repository: {MODEL_REPO}")
    print(f"Logs: {LOG_DIR}/triton_usage.log")
    print(f"\nAPI Keys for demo:")
    for key, user in API_KEYS.items():
        print(f"  {user}: {key}")
    print(f"\nEndpoints:")
    print(f"  Health: curl http://localhost:8000/api/v1/triton-inference/health")
    print(f"  List models: curl -H 'Authorization: Bearer demo-key-admin' http://localhost:8000/api/v1/triton-inference/v2/models")
    print(f"  Inference: POST http://localhost:8000/api/v1/triton-inference/v2/models/{{model_name}}/infer")
    print(f"\nRay Dashboard: http://<head-ip>:8265")
    print(f"Triton Metrics: http://localhost:8003/metrics")
    print("="*70 + "\n")
