# Ray Serve Object Detection Endpoints

This directory contains Ray Serve deployment for Object Detection models (YOLO, transformers, etc.) with autoscaling and dynamic input discovery.

## Features

- ✅ **Autoscaling with scale-to-zero** - Models scale down after 5 minutes of inactivity
- ✅ **Dynamic schema discovery** - Automatically discovers model inputs and outputs
- ✅ **GPU support** - Automatic GPU detection and usage
- ✅ **Multiple model types** - Supports YOLO (Ultralytics) and Transformers models
- ✅ **Compute metrics** - CPU, GPU, and memory usage in API responses
- ✅ **Polling support** - HTTP 202 responses when scaling up from zero

## Quick Start

### Deploy YOLO Model

```bash
cd ray-object-detection-endpoints

# Deploy default YOLO model (yolov8n.pt - will download automatically)
./deploy_object_detection.sh

# Or deploy a specific model
./deploy_object_detection.sh "yolov8s.pt" "my-yolo-app"

# Or deploy a HuggingFace transformers model
./deploy_object_detection.sh "facebook/detr-resnet-50" "detr-app"
```

### Using Python directly

```bash
/opt/ray/bin/python deploy_object_detection_v2.py \
  --model "yolov8n.pt" \
  --app-name "yolo-detection"
```

## Supported Models

- **YOLO models** (via Ultralytics): `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- **Transformers models**: Any HuggingFace object detection model (e.g., `facebook/detr-resnet-50`)

## API Endpoints

Once deployed, the following endpoints are available:

- **Schema Discovery**: `GET /{app-name}/v1/schema` - Discover model inputs and capabilities
- **Health Check**: `GET /{app-name}/v1/health` - Check if model is ready
- **List Models**: `GET /{app-name}/v1/models` - List available models
- **Detect Objects**: `POST /{app-name}/v1/detect` - Detect objects in image

## Example Usage

### Get Schema

```bash
curl http://<head-node-ip>:8000/object-detection-yolov8n-pt/v1/schema
```

### Detect Objects in Image

```bash
# First, encode your image to base64
IMAGE_B64=$(base64 -w 0 your_image.jpg)

# Then send detection request
curl -X POST http://<head-node-ip>:8000/object-detection-yolov8n-pt/v1/detect \
  -H 'Content-Type: application/json' \
  -d "{
    \"image\": \"data:image/jpeg;base64,$IMAGE_B64\",
    \"confidence\": 0.25,
    \"iou_threshold\": 0.45
  }"
```

### Using Python to encode and detect

```python
import base64
import requests
import json

# Read and encode image
with open("image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://<head-node-ip>:8000/object-detection-yolov8n-pt/v1/detect",
    json={
        "image": f"data:image/jpeg;base64,{img_data}",
        "confidence": 0.25,
        "iou_threshold": 0.45
    }
)

result = response.json()
print(f"Found {result['detection_count']} objects")
for det in result['detections']:
    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
```

## Response Format

```json
{
  "status": "success",
  "model": "yolov8n-pt",
  "detections": [
    {
      "class": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "detection_count": 1,
  "parameters": {
    "confidence": 0.25,
    "iou_threshold": 0.45
  },
  "compute_metrics": {
    "query_duration_seconds": 0.123,
    "replica_gpu_memory_allocated_mb": 2048,
    ...
  }
}
```

## Model Loading

- **YOLO models**: Automatically downloaded from Ultralytics if not found locally
- **Transformers models**: Loaded from HuggingFace (cached in `/data/models`)
- Models load on GPU if available, fall back to CPU otherwise
- Scale to zero after 5 minutes of inactivity

## Parameters

- **confidence**: Confidence threshold (0.0-1.0) - default: 0.25 for YOLO, 0.5 for transformers
- **iou_threshold**: IoU threshold for NMS (0.0-1.0) - default: 0.45 for YOLO, 0.5 for transformers
- **image_size**: Input image size (YOLO only) - default: 640
- **max_det**: Maximum detections (YOLO only) - default: 300
- **agnostic_nms**: Class-agnostic NMS (YOLO only) - default: false

## Troubleshooting

- **Model not loading**: Check logs for errors, ensure dependencies are installed
- **GPU not available**: Model will fall back to CPU (slower)
- **Scale-to-zero**: First request may take 60+ seconds while model loads
- **Image format**: Supports JPEG, PNG, etc. via PIL/Pillow
