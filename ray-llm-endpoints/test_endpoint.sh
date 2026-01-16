#!/bin/bash
# Test script for GPT-OSS LLM endpoint

set -e

# Default values
HEAD_NODE_IP="${HEAD_NODE_IP:-172.26.92.232}"
RAY_SERVE_PORT="${RAY_SERVE_PORT:-8001}"
ENDPOINT="${ENDPOINT:-http://${HEAD_NODE_IP}:${RAY_SERVE_PORT}/v1}"

echo "🧪 Testing GPT-OSS LLM Endpoint"
echo "=============================="
echo "Endpoint: ${ENDPOINT}"
echo ""

# Test 1: Health check (if available)
echo "=== Testing Health Check ==="
curl -s "${ENDPOINT}/health" || echo "⚠️  Health endpoint not available (this is OK)"
echo ""

# Test 2: List models
echo "=== Testing List Models ==="
curl -s "${ENDPOINT}/models" | python3 -m json.tool || {
    echo "⚠️  Models endpoint not available"
}
echo ""

# Test 3: Chat completion
echo "=== Testing Chat Completion ==="
curl -s -X POST "${ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-gpt-oss",
    "messages": [
      {"role": "user", "content": "How many r'\''s in strawberry?"}
    ],
    "stream": false,
    "max_tokens": 50
  }' | python3 -m json.tool || {
    echo "⚠️  Chat completion failed"
}
echo ""

# Test 4: Streaming chat completion
echo "=== Testing Streaming Chat Completion ==="
echo "(This will stream tokens as they're generated)"
curl -s -X POST "${ENDPOINT}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-gpt-oss",
    "messages": [
      {"role": "user", "content": "Say hello in one word"}
    ],
    "stream": true,
    "max_tokens": 10
  }' | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        echo "$line"
    fi
done
echo ""

echo "✅ Testing complete!"
echo ""
echo "📝 Example Python client:"
echo "from openai import OpenAI"
echo "client = OpenAI(base_url=\"${ENDPOINT}\", api_key=\"not-needed\")"
echo "response = client.chat.completions.create("
echo "    model=\"my-gpt-oss\","
echo "    messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]"
echo ")"
