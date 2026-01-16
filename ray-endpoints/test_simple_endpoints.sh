#!/bin/bash
# Test script for simple ONNX endpoints

set -e

# Default values
HEAD_NODE_IP="${HEAD_NODE_IP:-172.26.92.232}"
RAY_SERVE_PORT="${RAY_SERVE_PORT:-8001}"
BASE_URL="http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}/classify"

echo "🧪 Testing Simple ONNX Endpoints"
echo "================================"
echo "Base URL: ${BASE_URL}"
echo ""

# Test image file (adjust path as needed)
IMAGE_FILE="${IMAGE_FILE:-boot.jpg}"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "⚠️  Image file not found: $IMAGE_FILE"
    echo "   Using a test request without image data"
    IMAGE_FILE=""
fi

# Test 1: Root endpoint (service info)
echo "=== Test 1: Root Endpoint (GET /) ==="
curl -s "${BASE_URL}/" | python3 -m json.tool || echo "⚠️  Failed"
echo ""

# Test 2: List models
echo "=== Test 2: List Models (GET /models) ==="
curl -s "${BASE_URL}/models" | python3 -m json.tool || echo "⚠️  Failed"
echo ""

# Test 3: Classify with ResNet50 (if image available)
if [ -f "$IMAGE_FILE" ]; then
    echo "=== Test 3: Classify with ResNet50 (POST /classify?model=resnet50) ==="
    curl -s -X POST "${BASE_URL}/classify?model=resnet50" \
         -H "Content-Type: image/jpeg" \
         --data-binary @"$IMAGE_FILE" | python3 -m json.tool || echo "⚠️  Failed"
    echo ""
    
    # Test 4: Dedicated ResNet50 endpoint
    echo "=== Test 4: Dedicated ResNet50 Endpoint (POST /resnet50) ==="
    curl -s -X POST "${BASE_URL}/resnet50" \
         -H "Content-Type: image/jpeg" \
         --data-binary @"$IMAGE_FILE" | python3 -m json.tool || echo "⚠️  Failed"
    echo ""
    
    # Test 5: MobileNetV2
    echo "=== Test 5: MobileNetV2 Endpoint (POST /mobilenetv2) ==="
    curl -s -X POST "${BASE_URL}/mobilenetv2" \
         -H "Content-Type: image/jpeg" \
         --data-binary @"$IMAGE_FILE" | python3 -m json.tool || echo "⚠️  Failed"
    echo ""
else
    echo "⚠️  Skipping image classification tests (no image file)"
    echo "   Set IMAGE_FILE environment variable to test with an image"
fi

echo "✅ Testing complete!"
echo ""
echo "📝 Quick Reference:"
echo "   Service info: curl ${BASE_URL}/"
echo "   List models: curl ${BASE_URL}/models"
echo "   Classify: curl -X POST \"${BASE_URL}/classify?model=resnet50\" \\"
echo "             -H \"Content-Type: image/jpeg\" --data-binary @image.jpg"
