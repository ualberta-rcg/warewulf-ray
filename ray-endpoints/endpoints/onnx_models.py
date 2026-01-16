"""
Ray Serve endpoints for ONNX models served via Triton
"""

import sys
import os

# Ensure we can import Ray (from /opt/ray in Docker container)
if "/opt/ray/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

from ray import serve
from typing import Dict, Any, Optional
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


@serve.deployment(
    name="onnx_model_proxy",
    num_replicas=1,
)
class ONNXModelProxy:
    """Proxy endpoint for ONNX models served by Triton"""
    
    def __init__(self):
        """Initialize Triton client"""
        self.client = tritonhttp.InferenceServerClient(url=TRITON_SERVER_URL)
        self.models = {}
        
        # Verify Triton server is available
        try:
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server at {TRITON_SERVER_URL} is not live")
            if not self.client.is_server_ready():
                raise RuntimeError(f"Triton server at {TRITON_SERVER_URL} is not ready")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Triton server: {e}")
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load available model metadata"""
        try:
            models = self.client.get_model_repository_index()
            for model in models:
                if model.get("name") in AVAILABLE_MODELS:
                    self.models[model["name"]] = model
                    print(f"✅ Loaded model: {model['name']}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load model repository index: {e}")
    
    def _preprocess_image(self, image_data: bytes) -> np.ndarray:
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
    
    async def __call__(self, request) -> Dict[str, Any]:
        """Handle inference request"""
        try:
            # Get model name and input data
            model_name = request.query_params.get("model", "resnet50")
            
            if model_name not in AVAILABLE_MODELS:
                return {
                    "error": f"Model '{model_name}' not available",
                    "available_models": AVAILABLE_MODELS
                }
            
            # Get image data
            if hasattr(request, "json"):
                data = await request.json()
                image_b64 = data.get("image")
                if image_b64:
                    image_data = base64.b64decode(image_b64)
                else:
                    return {"error": "No image data provided"}
            else:
                # Assume raw image bytes
                image_data = await request.body()
            
            # Preprocess image
            input_data = self._preprocess_image(image_data)
            
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
            
        except Exception as e:
            return {
                "error": str(e),
                "type": type(e).__name__
            }
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return {
            "available_models": AVAILABLE_MODELS,
            "triton_server": TRITON_SERVER_URL,
            "models_loaded": list(self.models.keys())
        }


@serve.deployment(
    name="resnet50",
    num_replicas=1,
)
class ResNet50Endpoint(ONNXModelProxy):
    """Dedicated ResNet50 endpoint"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "resnet50"
    
    async def __call__(self, request):
        # Override to always use resnet50
        request.query_params = {"model": "resnet50"}
        return await super().__call__(request)


@serve.deployment(
    name="mobilenetv2",
    num_replicas=1,
)
class MobileNetV2Endpoint(ONNXModelProxy):
    """Dedicated MobileNetV2 endpoint"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "mobilenetv2"
    
    async def __call__(self, request):
        request.query_params = {"model": "mobilenetv2"}
        return await super().__call__(request)


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


if __name__ == "__main__":
    # For testing
    from ray import serve
    import ray
    
    ray.init()
    serve.start(detached=True)
    deploy()
