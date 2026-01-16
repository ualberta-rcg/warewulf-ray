"""
Simple ONNX model endpoints following Ray docs pattern
Based on: https://docs.ray.io/en/latest/serve/tutorials/triton-server-integration.html
Uses HTTP client since tritonserver Python API is not available
"""

import sys
import os

# Ensure we can import Ray (from /opt/ray in Docker container)
if "/opt/ray/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/ray/bin:" + os.environ.get("PATH", "")

from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Dict, Any
import numpy as np
from PIL import Image
import io

# Triton HTTP client (since Python API not available)
try:
    import tritonclient.http as tritonhttp
    from tritonclient import http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️  tritonclient not available")

# Configuration
MODEL_REPOSITORY = "/data/ray-triton/model_repository"
TRITON_URL = "localhost:8000"

# Available ONNX models
AVAILABLE_MODELS = [
    "mobilenetv2",
    "resnet18",
    "resnet50",
    "shufflenet",
    "squeezenet",
    "vgg16",
]


def preprocess_image(image_data: bytes) -> np.ndarray:
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


app = FastAPI()


@serve.deployment(
    name="onnx_classifier",
    num_replicas=1,
    ray_actor_options={
        "runtime_env": {
            "pip": ["Pillow>=10.0.0", "numpy>=1.24.0,<2.0.0", "onnxruntime>=1.16.0"]
        }
    }
)
@serve.ingress(app)
class ONNXClassifier:
    """Simple ONNX classifier following Ray docs pattern"""
    
    def __init__(self):
        """Initialize Triton client"""
        if not TRITON_AVAILABLE:
            raise ImportError("tritonclient not available. Install with: pip install tritonclient[all]")
        
        self.client = tritonhttp.InferenceServerClient(url=TRITON_URL)
        
        # Check if Triton server is available
        try:
            if not self.client.is_server_live():
                print(f"⚠️  Triton server at {TRITON_URL} is not live")
            elif not self.client.is_server_ready():
                print(f"⚠️  Triton server at {TRITON_URL} is not ready")
            else:
                print(f"✅ Connected to Triton server at {TRITON_URL}")
        except Exception as e:
            print(f"⚠️  Could not connect to Triton: {e}")
            print(f"   Make sure Triton is running on {TRITON_URL}")
    
    @app.get("/")
    def root(self) -> Dict[str, Any]:
        """Root endpoint - list available models"""
        return {
            "service": "ONNX Classifier",
            "available_models": AVAILABLE_MODELS,
            "triton_url": TRITON_URL,
            "status": "ready" if TRITON_AVAILABLE else "triton_client_not_available"
        }
    
    @app.get("/models")
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return {
            "models": AVAILABLE_MODELS,
            "count": len(AVAILABLE_MODELS)
        }
    
    @app.post("/classify")
    async def classify(self, request: Request) -> JSONResponse:
        """Classify image - model specified in query param"""
        try:
            # Get model name from query params
            model_name = request.query_params.get("model", "resnet50")
            
            if model_name not in AVAILABLE_MODELS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Model '{model_name}' not available",
                        "available_models": AVAILABLE_MODELS
                    }
                )
            
            # Get image data
            image_data = await request.body()
            if not image_data:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No image data provided"}
                )
            
            # Preprocess image
            input_data = preprocess_image(image_data)
            
            # Prepare Triton inputs
            inputs = [
                httpclient.InferInput("data", input_data.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_data)
            
            # Request output (try common output names)
            outputs = [
                httpclient.InferRequestedOutput("fc6")  # Common for classification
            ]
            
            # Perform inference
            response = self.client.infer(model_name, inputs, outputs=outputs)
            predictions = response.as_numpy("fc6")
            
            # Get top prediction
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return JSONResponse({
                "model": model_name,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_5_predictions": [
                    {"class": int(i), "score": float(score)}
                    for i, score in enumerate(np.argsort(predictions[0])[-5:][::-1])
                ]
            })
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "type": type(e).__name__
                }
            )
    
    @app.post("/resnet50")
    async def resnet50(self, request: Request) -> JSONResponse:
        """Dedicated ResNet50 endpoint"""
        # Call classify with model name hardcoded
        try:
            image_data = await request.body()
            if not image_data:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No image data provided"}
                )
            
            input_data = preprocess_image(image_data)
            inputs = [httpclient.InferInput("data", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [httpclient.InferRequestedOutput("fc6")]
            
            response = self.client.infer("resnet50", inputs, outputs=outputs)
            predictions = response.as_numpy("fc6")
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return JSONResponse({
                "model": "resnet50",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_5_predictions": [
                    {"class": int(i), "score": float(score)}
                    for i, score in enumerate(np.argsort(predictions[0])[-5:][::-1])
                ]
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type(e).__name__}
            )
    
    @app.post("/mobilenetv2")
    async def mobilenetv2(self, request: Request) -> JSONResponse:
        """Dedicated MobileNetV2 endpoint"""
        try:
            image_data = await request.body()
            if not image_data:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No image data provided"}
                )
            
            input_data = preprocess_image(image_data)
            inputs = [httpclient.InferInput("data", input_data.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [httpclient.InferRequestedOutput("fc6")]
            
            response = self.client.infer("mobilenetv2", inputs, outputs=outputs)
            predictions = response.as_numpy("fc6")
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            
            return JSONResponse({
                "model": "mobilenetv2",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top_5_predictions": [
                    {"class": int(i), "score": float(score)}
                    for i, score in enumerate(np.argsort(predictions[0])[-5:][::-1])
                ]
            })
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type(e).__name__}
            )


def deploy():
    """Deploy the ONNX classifier"""
    print("📦 Deploying simple ONNX classifier endpoint...")
    
    serve.run(
        ONNXClassifier.bind(),
        name="onnx_classifier",
        route_prefix="/classify"
    )
    
    print("✅ Deployed ONNX classifier at /classify")
    print("\n📝 Available endpoints:")
    print("  GET  /classify/          - Service info and available models")
    print("  GET  /classify/models    - List models")
    print("  POST /classify/classify?model=resnet50 - Classify with specified model")
    print("  POST /classify/resnet50  - Classify with ResNet50")
    print("  POST /classify/mobilenetv2 - Classify with MobileNetV2")
    print("\n📝 Example curl commands:")
    print('  curl -X POST "http://<head-node-ip>:8001/classify/classify?model=resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')
    print('\n  curl -X POST "http://<head-node-ip>:8001/classify/resnet50" \\')
    print('       -H "Content-Type: image/jpeg" \\')
    print('       --data-binary @image.jpg')


if __name__ == "__main__":
    deploy()
