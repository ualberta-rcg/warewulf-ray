"""
Base class for all model endpoints
"""
from abc import ABC, abstractmethod
from ray.serve.config import AutoscalingConfig
from fastapi import FastAPI
from typing import Dict, Any
import time

class BaseEndpoint(ABC):
    """Base class for all model endpoints"""
    
    # Override these in your endpoint
    ROUTE_PREFIX: str = None  # e.g., "/api/v1/stable-diffusion"
    DEPLOYMENT_NAME: str = None  # e.g., "stable-diffusion"
    MIN_REPLICAS: int = 0
    MAX_REPLICAS: int = 10
    DOWNSCALE_DELAY_S: int = 60
    
    def __init__(self):
        """Initialize model - override this"""
        print(f"Loading model for {self.DEPLOYMENT_NAME}...")
        self.model = self.load_model()
        self.ready = True
        print(f"✓ {self.DEPLOYMENT_NAME} ready!")
    
    @abstractmethod
    def load_model(self):
        """Load your model - must implement"""
        pass
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference - must implement"""
        pass
    
    def get_autoscaling_config(self) -> AutoscalingConfig:
        """Get autoscaling config - can override"""
        return AutoscalingConfig(
            min_replicas=self.MIN_REPLICAS,
            max_replicas=self.MAX_REPLICAS,
            target_num_ongoing_requests_per_replica=1,
            downscale_delay_s=self.DOWNSCALE_DELAY_S,
            upscale_delay_s=0,
        )
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app with standard endpoints - override for custom"""
        app = FastAPI(title=self.DEPLOYMENT_NAME)
        
        @app.get("/health")
        async def health():
            return {
                "status": "ready" if self.ready else "loading",
                "model": self.DEPLOYMENT_NAME
            }
        
        @app.post("/predict")
        async def predict(request):
            data = await request.json()
            result = self.predict(data)
            return result
        
        @app.get("/info")
        async def info():
            return {
                "name": self.DEPLOYMENT_NAME,
                "route": self.ROUTE_PREFIX,
                "ready": self.ready
            }
        
        return app
