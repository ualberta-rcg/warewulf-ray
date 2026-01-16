#!/bin/bash
# Script to check Ray Serve routes and endpoints
# This helps verify that endpoints are actually deployed and accessible

set -e

# Get head node IP from Ray or use default
HEAD_NODE_IP="${RAY_HEAD_NODE_IP:-172.26.92.232}"
RAY_SERVE_PORT=8000
DASHBOARD_PORT=8265

echo "🔍 Checking Ray Serve Routes and Endpoints"
echo "=========================================="
echo ""

# Method 1: Check Ray Serve status
echo "=== Method 1: Ray Serve Status ==="
if command -v ray &> /dev/null; then
    ray serve status 2>/dev/null || echo "  (ray serve status not available)"
else
    echo "  (ray command not found)"
fi
echo ""

# Method 2: Check via Dashboard API
echo "=== Method 2: Dashboard API ==="
echo "Checking http://${HEAD_NODE_IP}:${DASHBOARD_PORT}/api/serve/applications/"
if command -v curl &> /dev/null; then
    response=$(curl -s -w "\nHTTP_CODE:%{http_code}" "http://${HEAD_NODE_IP}:${DASHBOARD_PORT}/api/serve/applications/" 2>/dev/null || echo "HTTP_CODE:000")
    http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_CODE/d')
    
    if [ "$http_code" = "200" ]; then
        echo "✅ Dashboard API accessible"
        if command -v jq &> /dev/null; then
            echo "$body" | jq '.' 2>/dev/null || echo "$body"
        else
            echo "$body"
        fi
    else
        echo "❌ Dashboard API not accessible (HTTP $http_code)"
        echo "   Make sure Ray dashboard is running on port ${DASHBOARD_PORT}"
    fi
else
    echo "  (curl not found)"
fi
echo ""

# Method 3: Test endpoints directly
echo "=== Method 3: Testing Endpoints ==="
ENDPOINTS=("/onnx" "/resnet50" "/mobilenetv2" "/stable-diffusion")

for endpoint in "${ENDPOINTS[@]}"; do
    echo -n "  Testing ${endpoint}... "
    if command -v curl &> /dev/null; then
        # Try a simple GET request (some endpoints might not support GET, but we'll see)
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 2 \
            "http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}${endpoint}" 2>/dev/null || echo "000")
        
        if [ "$http_code" = "200" ] || [ "$http_code" = "405" ] || [ "$http_code" = "422" ]; then
            echo "✅ Endpoint exists (HTTP $http_code)"
        elif [ "$http_code" = "404" ]; then
            echo "❌ Endpoint not found (404)"
        elif [ "$http_code" = "000" ]; then
            echo "❌ Cannot connect (check if Ray Serve is running)"
        else
            echo "⚠️  Got HTTP $http_code"
        fi
    else
        echo "  (curl not found)"
    fi
done
echo ""

# Method 4: Check what's listening on port 8000
echo "=== Method 4: Port 8000 Status ==="
if command -v netstat &> /dev/null; then
    netstat -tlnp 2>/dev/null | grep ":${RAY_SERVE_PORT} " || echo "  (Nothing listening on port ${RAY_SERVE_PORT} or netstat failed)"
elif command -v ss &> /dev/null; then
    ss -tlnp 2>/dev/null | grep ":${RAY_SERVE_PORT} " || echo "  (Nothing listening on port ${RAY_SERVE_PORT} or ss failed)"
else
    echo "  (netstat/ss not available)"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "Head Node IP: ${HEAD_NODE_IP}"
echo "Ray Serve Port: ${RAY_SERVE_PORT}"
echo "Dashboard Port: ${DASHBOARD_PORT}"
echo ""
echo "Expected endpoints:"
for endpoint in "${ENDPOINTS[@]}"; do
    echo "  - http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}${endpoint}"
done
echo ""
echo "Dashboard: http://${HEAD_NODE_IP}:${DASHBOARD_PORT}"
