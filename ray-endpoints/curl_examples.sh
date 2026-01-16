#!/bin/bash
# Example curl commands for testing Ray Serve endpoints
# Make sure endpoints are deployed first: ./deploy.sh

RAY_SERVE_URL="${RAY_SERVE_URL:-http://localhost:8000}"
IMAGE_FILE="${IMAGE_FILE:-test_image.jpg}"

echo "Testing Ray Serve endpoints at $RAY_SERVE_URL"
echo "Using image file: $IMAGE_FILE"
echo ""

# Test 1: Proxy endpoint with ResNet50
echo "=== Test 1: Proxy endpoint with ResNet50 ==="
curl -X POST "${RAY_SERVE_URL}/onnx?model=resnet50" \
     -H "Content-Type: image/jpeg" \
     --data-binary "@${IMAGE_FILE}" \
     -w "\nHTTP Status: %{http_code}\n" \
     | jq '.' 2>/dev/null || cat
echo ""

# Test 2: Proxy endpoint with MobileNetV2
echo "=== Test 2: Proxy endpoint with MobileNetV2 ==="
curl -X POST "${RAY_SERVE_URL}/onnx?model=mobilenetv2" \
     -H "Content-Type: image/jpeg" \
     --data-binary "@${IMAGE_FILE}" \
     -w "\nHTTP Status: %{http_code}\n" \
     | jq '.' 2>/dev/null || cat
echo ""

# Test 3: Dedicated ResNet50 endpoint
echo "=== Test 3: Dedicated ResNet50 endpoint ==="
curl -X POST "${RAY_SERVE_URL}/resnet50" \
     -H "Content-Type: image/jpeg" \
     --data-binary "@${IMAGE_FILE}" \
     -w "\nHTTP Status: %{http_code}\n" \
     | jq '.' 2>/dev/null || cat
echo ""

# Test 4: Dedicated MobileNetV2 endpoint
echo "=== Test 4: Dedicated MobileNetV2 endpoint ==="
curl -X POST "${RAY_SERVE_URL}/mobilenetv2" \
     -H "Content-Type: image/jpeg" \
     --data-binary "@${IMAGE_FILE}" \
     -w "\nHTTP Status: %{http_code}\n" \
     | jq '.' 2>/dev/null || cat
echo ""

# Test 5: List available models (if endpoint supports it)
echo "=== Test 5: Check endpoint health ==="
curl -X GET "${RAY_SERVE_URL}/onnx?model=list" \
     -w "\nHTTP Status: %{http_code}\n" \
     | jq '.' 2>/dev/null || cat
echo ""

echo "Done! Check the responses above for inference results."
