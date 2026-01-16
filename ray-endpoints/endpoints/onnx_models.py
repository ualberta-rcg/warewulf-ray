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

# Triton imports - try Python API first (embedded), fallback to HTTP client
try:
    import tritonserver
    TRITON_PYTHON_API_AVAILABLE = True
except ImportError:
    TRITON_PYTHON_API_AVAILABLE = False
    try:
        import tritonclient.http as tritonhttp
        from tritonclient import http as httpclient
    except ImportError:
        raise ImportError("Neither tritonserver nor tritonclient available. Install with: pip install nvidia-pytriton or tritonclient[all]")

# Configuration
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
    
    def __init__(self, use_embedded_triton: bool = True):
        """Initialize Triton - use embedded server if available, else HTTP client"""
        self.use_embedded = use_embedded_triton and TRITON_PYTHON_API_AVAILABLE
        self.triton_server = None
        self.client = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure Triton is initialized (lazy initialization)"""
        if self._initialized:
            return
        
        if self.use_embedded:
            # Use embedded Triton server (starts inside Ray Serve deployment)
            # Following official Ray docs: https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html
            if self.triton_server is None:
                # model_repository can be a string or list
                self.triton_server = tritonserver.Server(
                    model_repository=[MODEL_REPOSITORY],  # Use list as per Ray docs
                    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
                    log_info=False,
                )
                self.triton_server.start(wait_until_ready=True)
                print(f"✅ Embedded Triton server started with models from {MODEL_REPOSITORY}")
            self._initialized = True
        else:
            # Fallback to HTTP client (external Triton server)
            if self.client is None:
                self.client = tritonhttp.InferenceServerClient(url="localhost:8000")
            
            try:
                if not self.client.is_server_live():
                    raise RuntimeError("Triton server at localhost:8000 is not live. Please start Triton server first.")
                if not self.client.is_server_ready():
                    raise RuntimeError("Triton server at localhost:8000 is not ready.")
                self._initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to connect to Triton server: {e}. Make sure Triton is running.")
    
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
        # Ensure Triton is initialized
        self._ensure_initialized()
        
        # Preprocess image
        input_data = self.preprocess_image(image_data)
        
        if self.use_embedded:
            # Use embedded Triton Python API (following Ray docs pattern)
            # Check if model is ready, load if needed
            if not self.triton_server.model(model_name).ready():
                self.triton_server.load(model_name)
            
            model = self.triton_server.model(model_name)
            
            # Perform inference - input name is typically "data" for ONNX models
            # Format: inputs={"input_name": [[batch_data]]}
            # Note: The double list is for batching - outer list is batch dimension
            predictions = None
            for response in model.infer(inputs={"data": [[input_data]]}):
                # Get output - ONNX models typically have output named "fc6" or similar
                # Try common output names first
                output_names = list(response.outputs.keys())
                if not output_names:
                    raise RuntimeError("No outputs found in Triton response")
                
                # Try "fc6" first (common for classification models), then first available
                output_name = "fc6" if "fc6" in output_names else output_names[0]
                output_tensor = response.outputs[output_name]
                
                # Convert from DLPack to numpy (as per Ray docs)
                predictions = np.from_dlpack(output_tensor).squeeze()
            
            if predictions is None:
                raise RuntimeError("No predictions received from Triton")
            
            # Handle prediction shape - should be 1D array of class probabilities
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            predicted_class = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            
            return {
                "model": model_name,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "predictions": predictions.tolist()[:10]
            }
        else:
            # Use HTTP client (external Triton)
            inputs = [
                httpclient.InferInput("data", input_data.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            outputs = [
                httpclient.InferRequestedOutput("fc6")
            ]
            
            response = self.client.infer(model_name, inputs, outputs=outputs)
            predictions = response.as_numpy("fc6")
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return {
                "model": model_name,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "predictions": predictions[0].tolist()[:10]
            }


@serve.deployment(
    name="onnx_model_proxy",
    num_replicas=1,
    ray_actor_options={
        "runtime_env": {
            "pip": ["Pillow>=10.0.0", "numpy>=1.24.0,<2.0.0", "onnxruntime>=1.16.0"]
        }
    }
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
    ray_actor_options={
        "runtime_env": {
            "pip": ["Pillow>=10.0.0", "numpy>=1.24.0,<2.0.0", "onnxruntime>=1.16.0"]
        }
    }
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
    ray_actor_options={
        "runtime_env": {
            "pip": ["Pillow>=10.0.0", "numpy>=1.24.0,<2.0.0", "onnxruntime>=1.16.0"]
        }
    }
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
    print("\n📝 Example curl commands (use your head node IP, e.g., 172.26.92.232):")
    print("  # Use proxy endpoint with model parameter")
    print('  curl -X POST "http://<head-node-ip>:8001/onnx?model=resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')
    print("\n  # Use dedicated ResNet50 endpoint")
    print('  curl -X POST "http://<head-node-ip>:8001/resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')
    print("\n  # Use dedicated MobileNetV2 endpoint")
    print('  curl -X POST "http://<head-node-ip>:8001/mobilenetv2" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')


if __name__ == "__main__":
    # For testing
    from ray import serve
    import ray
    
    ray.init()
    serve.start(detached=True)
    deploy()
