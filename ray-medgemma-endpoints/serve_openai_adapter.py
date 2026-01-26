"""
OpenAI API Adapter for MedGemma
Proxies OpenAI Chat Completions format requests to existing MedGemma endpoint
Does NOT modify the original MedGemma code - just queries it via HTTP
"""

import json
import uuid
import httpx
import base64
import io
from typing import Any, Dict, List, Optional
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve


def create_deployment(medgemma_base_url: str, model_name: str = "medgemma-27b-it"):
    """Create a deployment that adapts OpenAI format to MedGemma format"""
    
    @serve.deployment(
        name="openai-adapter",
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_num_ongoing_requests_per_replica": 2,
        },
        ray_actor_options={
            "runtime_env": {
                "pip": [
                    "httpx>=0.24.0",
                    "fastapi>=0.100.0",
                    "pillow>=9.0.0",
                ]
            }
        }
    )
    class OpenAIAdapter:
        def __init__(self, medgemma_base_url: str = medgemma_base_url, model_name: str = model_name):
            self.medgemma_url = medgemma_base_url
            self.model_name = model_name
            print(f"✅ OpenAI adapter initialized")
            print(f"   Proxying to MedGemma: {self.medgemma_url}")
            print(f"   Model name: {self.model_name}")
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            if path.endswith("/v1/chat/completions"):
                return await self.handle_chat_completions(request)
            elif path.endswith("/v1/models"):
                return await self.handle_models(request)
            elif path.startswith("/v1/models/") and path.count("/") == 2:
                # /v1/models/{model_id}
                return await self.handle_model_info(request)
            else:
                return JSONResponse({"error": f"Unknown endpoint: {path}"}, status_code=404)
        
        async def handle_chat_completions(self, request: Request) -> JSONResponse:
            """Convert OpenAI Chat Completions → MedGemma generate"""
            try:
                # Get OpenAI format request
                openai_data = await request.json()
                
                # Check if streaming is requested
                stream = openai_data.get("stream", False)
                
                # Transform OpenAI messages to MedGemma format
                medgemma_messages = self._transform_messages(openai_data.get("messages", []))
                
                # Build MedGemma request
                medgemma_data = {
                    "messages": medgemma_messages,
                    "max_new_tokens": openai_data.get("max_tokens", 200),  # max_tokens → max_new_tokens
                    "temperature": openai_data.get("temperature"),
                    "top_p": openai_data.get("top_p"),
                    "do_sample": openai_data.get("temperature", 0) > 0 or openai_data.get("top_p") is not None,
                }
                # Remove None values
                medgemma_data = {k: v for k, v in medgemma_data.items() if v is not None}
                
                # Call existing MedGemma endpoint
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.medgemma_url}/generate",
                        json=medgemma_data
                    )
                    response.raise_for_status()
                    medgemma_result = response.json()
                
                # Handle streaming response
                if stream:
                    return await self._handle_streaming_response(medgemma_result, openai_data)
                
                # Transform response back to OpenAI format
                response_text = medgemma_result.get("response", "")
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                
                # Estimate token counts (rough approximation: 1 token ≈ 4 characters)
                prompt_tokens = sum(len(msg.get("content", "")) for msg in openai_data.get("messages", [])) // 4
                completion_tokens = len(response_text) // 4
                total_tokens = prompt_tokens + completion_tokens
                
                return JSONResponse({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(__import__("time").time()),
                    "model": openai_data.get("model", self.model_name),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                })
            except httpx.HTTPStatusError as e:
                return JSONResponse(
                    {"error": {"message": f"MedGemma endpoint error: {e.response.text}", "type": "api_error"}},
                    status_code=e.response.status_code
                )
            except Exception as e:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "internal_error"}},
                    status_code=500
                )
        
        async def _handle_streaming_response(self, medgemma_result: Dict, openai_data: Dict):
            """Handle streaming response (Server-Sent Events format)"""
            response_text = medgemma_result.get("response", "")
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            model_name = openai_data.get("model", self.model_name)
            
            # For streaming, we send the full response in chunks
            # In a real implementation, you'd want to stream token-by-token
            # For now, we'll send it in reasonable chunks
            chunk_size = 20  # characters per chunk
            chunks = [response_text[i:i+chunk_size] for i in range(0, len(response_text), chunk_size)]
            
            async def generate():
                # Send initial chunk
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(__import__("time").time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None if i < len(chunks) - 1 else "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send final done message
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        def _transform_messages(self, openai_messages: List[Dict]) -> List[Dict]:
            """Transform OpenAI messages format to MedGemma format"""
            medgemma_messages = []
            
            for msg in openai_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle different content types
                if isinstance(content, str):
                    # Simple text content
                    medgemma_content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    # Multi-modal content (text + images)
                    medgemma_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                medgemma_content.append({"type": "text", "text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                # OpenAI format: {"type": "image_url", "image_url": {"url": "data:..."}}
                                image_url = item.get("image_url", {})
                                if isinstance(image_url, dict):
                                    url = image_url.get("url", "")
                                else:
                                    url = str(image_url)
                                
                                # Send image as base64 data URI string (MedGemma will decode it)
                                # Ensure it's in data URI format
                                if not url.startswith("data:"):
                                    url = f"data:image/png;base64,{url}"
                                medgemma_content.append({"type": "image", "image": url})
                        else:
                            # Fallback: treat as text
                            medgemma_content.append({"type": "text", "text": str(item)})
                else:
                    # Fallback: convert to string
                    medgemma_content = [{"type": "text", "text": str(content)}]
                
                medgemma_messages.append({
                    "role": role,
                    "content": medgemma_content
                })
            
            return medgemma_messages
        
        
        async def handle_models(self, request: Request) -> JSONResponse:
            """List available models (OpenAI format)"""
            return JSONResponse({
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "medgemma",
                    "permission": [],
                    "root": self.model_name,
                    "parent": None
                }]
            })
        
        async def handle_model_info(self, request: Request) -> JSONResponse:
            """Get model info (OpenAI format)"""
            path = request.url.path
            model_id = path.split("/")[-1]
            
            if model_id == self.model_name:
                return JSONResponse({
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "medgemma",
                    "permission": [],
                    "root": self.model_name,
                    "parent": None
                })
            else:
                return JSONResponse(
                    {"error": {"message": f"Model {model_id} not found", "type": "invalid_request_error"}},
                    status_code=404
                )
    
    return OpenAIAdapter


def create_app(medgemma_base_url: str, model_name: str = "medgemma-27b-it"):
    """Create the adapter app"""
    deployment_class = create_deployment(medgemma_base_url, model_name)
    return deployment_class.bind(medgemma_base_url=medgemma_base_url, model_name=model_name)
