#!/bin/bash
# Quick test script for Ray Serve endpoints
# Tests if endpoints are accessible and responding

HEAD_NODE_IP="${HEAD_NODE_IP:-172.26.92.232}"
RAY_SERVE_PORT="${RAY_SERVE_PORT:-8001}"  # Ray Serve on 8001, Triton on 8000
IMAGE_FILE="${IMAGE_FILE:-boot.jpg}"

echo "🧪 Testing Ray Serve Endpoints"
echo "=============================="
echo "Head Node: ${HEAD_NODE_IP}"
echo "Port: ${RAY_SERVE_PORT}"
echo ""

# Check if image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo "⚠️  Image file '$IMAGE_FILE' not found"
    echo "   Using a simple GET request instead (endpoints may require POST)"
    USE_GET=true
else
    USE_GET=false
    echo "Using image file: $IMAGE_FILE"
fi
echo ""

# Test endpoints
ENDPOINTS=(
    "/onnx?model=resnet50"
    "/resnet50"
    "/mobilenetv2"
    "/stable-diffusion"
)

for endpoint in "${ENDPOINTS[@]}"; do
    echo "=== Testing ${endpoint} ==="
    
    if [ "$USE_GET" = true ]; then
        # Simple GET request to check if endpoint exists
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
            --max-time 5 \
            "http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}${endpoint}" 2>/dev/null || echo "HTTP_CODE:000")
    else
        # POST request with image
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
            --max-time 10 \
            -X POST \
            -H "Content-Type: image/jpeg" \
            --data-binary "@${IMAGE_FILE}" \
            "http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}${endpoint}" 2>/dev/null || echo "HTTP_CODE:000")
    fi
    
    http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_CODE/d')
    
    if [ "$http_code" = "200" ]; then
        echo "✅ Endpoint accessible (HTTP 200)"
        if command -v jq &> /dev/null && [ -n "$body" ]; then
            echo "$body" | jq '.' 2>/dev/null | head -10 || echo "$body" | head -5
        else
            echo "$body" | head -5
        fi
    elif [ "$http_code" = "405" ]; then
        echo "⚠️  Method not allowed (HTTP 405) - endpoint exists but needs POST"
    elif [ "$http_code" = "422" ] || [ "$http_code" = "400" ]; then
        echo "⚠️  Bad request (HTTP $http_code) - endpoint exists but request format wrong"
        echo "$body" | head -3
    elif [ "$http_code" = "404" ]; then
        echo "❌ Endpoint not found (HTTP 404)"
    elif [ "$http_code" = "000" ]; then
        echo "❌ Cannot connect - check if Ray Serve is running"
    else
        echo "⚠️  Got HTTP $http_code"
        echo "$body" | head -3
    fi
    echo ""
done

echo "=== Summary ==="
echo "If you see HTTP 200 or 405/422, endpoints are working!"
echo "HTTP 405/422 usually means the endpoint exists but needs correct request format."
echo ""
echo "To test with an image:"
echo "  curl -X POST \"http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}/resnet50\" \\"
echo "       -H \"Content-Type: image/jpeg\" \\"
echo "       --data-binary @${IMAGE_FILE}"
