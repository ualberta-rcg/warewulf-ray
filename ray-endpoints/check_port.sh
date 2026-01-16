#!/bin/bash
# Check what's listening on port 8000

echo "🔍 Checking what's listening on port 8000..."
echo ""

# Check with netstat
if command -v netstat &> /dev/null; then
    echo "=== netstat ==="
    netstat -tlnp 2>/dev/null | grep ":8000 " || echo "  Nothing found with netstat"
    echo ""
fi

# Check with ss
if command -v ss &> /dev/null; then
    echo "=== ss ==="
    ss -tlnp 2>/dev/null | grep ":8000 " || echo "  Nothing found with ss"
    echo ""
fi

# Check with lsof
if command -v lsof &> /dev/null; then
    echo "=== lsof ==="
    lsof -i :8000 2>/dev/null || echo "  Nothing found with lsof"
    echo ""
fi

# Try to connect
echo "=== Connection Test ==="
echo "Testing localhost:8000..."
curl -s -o /dev/null -w "localhost:8000 - HTTP %{http_code}\n" \
    --max-time 2 \
    http://localhost:8000/ 2>/dev/null || echo "localhost:8000 - Cannot connect"

echo ""
echo "Testing 0.0.0.0:8000..."
curl -s -o /dev/null -w "0.0.0.0:8000 - HTTP %{http_code}\n" \
    --max-time 2 \
    http://0.0.0.0:8000/ 2>/dev/null || echo "0.0.0.0:8000 - Cannot connect"

echo ""
echo "Testing 172.26.92.232:8000..."
curl -s -o /dev/null -w "172.26.92.232:8000 - HTTP %{http_code}\n" \
    --max-time 2 \
    http://172.26.92.232:8000/ 2>/dev/null || echo "172.26.92.232:8000 - Cannot connect"
