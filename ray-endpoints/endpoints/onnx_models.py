"""
Ray Serve endpoints for ONNX models served via Triton
Compatible with Ray Serve 2.53.0
"""

import sys
import os

# Ensure we can import Ray (from /opt/ray in Docker container)
if "/opt/ray/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

from ray import serve
from starlette.requests import Request
from typing import Dict, Any
import numpy as np
from PIL import Image
import io
import base64

# Triton client imports
try:
    import tritonclient.http as tritonhttp
    from tritonclient import http as httpclient
except ImportError:
    raise ImportError("tritonclient not installed. Install with: pip install tritonclient[all]")

# Configuration
TRITON_SERVER_URL = "localhost:8000"  # Triton server URL
MODEL_REPOSITORY = "/data/ray-triton/model_repository"

# Available ONNX models
AVAILABLE_MODELS = [
    "mobilenetv2",
    "resnet18",
    "resnet50",
    "shufflenet",
    "squeezenet",
    "vgg16",
]


class TritonModelHandler:
    """Shared handler for Triton model inference (not decorated)"""
    
    def __init__(self, triton_url: str = TRITON_SERVER_URL):
        """Initialize Triton client (lazy connection)"""
        self.triton_url = triton_url
        self.client = None
        self._connected = False
    
    def _ensure_connected(self):
        """Ensure Triton client is connected (lazy initialization)"""
        if self._connected and self.client is not None:
            return
        
        if self.client is None:
            self.client = tritonhttp.InferenceServerClient(url=self.triton_url)
        
        # Verify Triton server is available
        try:
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server at {self.triton_url} is not live. Please start Triton server first.")
            if not self.client.is_server_ready():
                raise RuntimeError(f"Triton server at {self.triton_url} is not ready. Please wait for it to initialize.")
            self._connected = True
        except Exception as e:
            self._connected = False
            raise RuntimeError(f"Failed to connect to Triton server at {self.triton_url}: {e}. "
                             f"Make sure Triton is running. See README for instructions.")
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for model input"""
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Resize to 224x224 (standard for these models)
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32)
        img_array = img_array / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def infer(self, model_name: str, image_data: bytes) -> Dict[str, Any]:
        """Perform inference on an image"""
        # Ensure connection to Triton
        self._ensure_connected()
        
        # Preprocess image
        input_data = self.preprocess_image(image_data)
        
        # Prepare Triton inputs
        inputs = [
            httpclient.InferInput("data", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)
        
        # Perform inference
        outputs = [
            httpclient.InferRequestedOutput("fc6")
        ]
        
        response = self.client.infer(model_name, inputs, outputs=outputs)
        
        # Get predictions
        predictions = response.as_numpy("fc6")
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return {
            "model": model_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "predictions": predictions[0].tolist()[:10]  # Top 10
        }


@serve.deployment(
    name="onnx_model_proxy",
    num_replicas=1,
)
class ONNXModelProxy:
    """Proxy endpoint for ONNX models served by Triton"""
    
    def __init__(self):
        """Initialize Triton client"""
        self.handler = TritonModelHandler()
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Handle inference request"""
        try:
            # Get model name from query params
            model_name = request.query_params.get("model", "resnet50")
            
            if model_name not in AVAILABLE_MODELS:
                return {
                    "error": f"Model '{model_name}' not available",
                    "available_models": AVAILABLE_MODELS
                }
            
            # Get image data
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                # JSON with base64 image
                data = await request.json()
                image_b64 = data.get("image", "")
                if not image_b64:
                    return {"error": "No image data provided in JSON"}
                image_data = base64.b64decode(image_b64)
            else:
                # Raw image bytes (most common for curl)
                image_data = await request.body()
                if not image_data:
                    return {"error": "No image data provided"}
            
            # Perform inference
            result = self.handler.infer(model_name, image_data)
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "type": type(e).__name__
            }


@serve.deployment(
    name="resnet50",
    num_replicas=1,
)
class ResNet50Endpoint:
    """Dedicated ResNet50 endpoint"""
    
    def __init__(self):
        self.handler = TritonModelHandler()
        self.model_name = "resnet50"
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Handle inference request for ResNet50"""
        try:
            # Get image data
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                data = await request.json()
                image_b64 = data.get("image", "")
                if not image_b64:
                    return {"error": "No image data provided in JSON"}
                image_data = base64.b64decode(image_b64)
            else:
                image_data = await request.body()
                if not image_data:
                    return {"error": "No image data provided"}
            
            result = self.handler.infer(self.model_name, image_data)
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "type": type(e).__name__
            }


@serve.deployment(
    name="mobilenetv2",
    num_replicas=1,
)
class MobileNetV2Endpoint:
    """Dedicated MobileNetV2 endpoint"""
    
    def __init__(self):
        self.handler = TritonModelHandler()
        self.model_name = "mobilenetv2"
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Handle inference request for MobileNetV2"""
        try:
            # Get image data
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                data = await request.json()
                image_b64 = data.get("image", "")
                if not image_b64:
                    return {"error": "No image data provided in JSON"}
                image_data = base64.b64decode(image_b64)
            else:
                image_data = await request.body()
                if not image_data:
                    return {"error": "No image data provided"}
            
            result = self.handler.infer(self.model_name, image_data)
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "type": type(e).__name__
            }


def deploy():
    """Deploy all ONNX model endpoints"""
    print("📦 Deploying ONNX model endpoints...")
    
    # Deploy proxy endpoint
    serve.run(ONNXModelProxy.bind(), name="onnx_model_proxy", route_prefix="/onnx")
    print("✅ Deployed ONNX model proxy at /onnx")
    
    # Deploy individual model endpoints
    serve.run(ResNet50Endpoint.bind(), name="resnet50", route_prefix="/resnet50")
    print("✅ Deployed ResNet50 endpoint at /resnet50")
    
    serve.run(MobileNetV2Endpoint.bind(), name="mobilenetv2", route_prefix="/mobilenetv2")
    print("✅ Deployed MobileNetV2 endpoint at /mobilenetv2")
    
    print("✅ All ONNX model endpoints deployed!")
    print("\n📝 Example curl commands:")
    print("  # Use proxy endpoint with model parameter")
    print('  curl -X POST "http://localhost:8000/onnx?model=resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')
    print("\n  # Use dedicated ResNet50 endpoint")
    print('  curl -X POST "http://localhost:8000/resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')
    print("\n  # Use dedicated MobileNetV2 endpoint")
    print('  curl -X POST "http://localhost:8000/mobilenetv2" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')


if __name__ == "__main__":
    # For testing
    from ray import serve
    import ray
    
    ray.init()
    serve.start(detached=True)
    deploy()
