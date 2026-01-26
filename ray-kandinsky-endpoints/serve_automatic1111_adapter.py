"""
Automatic1111 API Adapter for Kandinsky
Proxies Automatic1111 format requests to existing Kandinsky endpoint
Does NOT modify the original Kandinsky code - just queries it via HTTP
"""

import json
import httpx
from typing import Any, Dict
from fastapi import Request
from fastapi.responses import JSONResponse
from ray import serve


def create_deployment(kandinsky_base_url: str):
    """Create a deployment that adapts Automatic1111 format to Kandinsky format"""
    
    @serve.deployment(
        name="automatic1111-adapter",
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
                ]
            }
        }
    )
    class Automatic1111Adapter:
        def __init__(self, kandinsky_base_url: str = kandinsky_base_url):
            self.kandinsky_url = kandinsky_base_url
            print(f"✅ Automatic1111 adapter initialized")
            print(f"   Proxying to Kandinsky: {self.kandinsky_url}")
        
        async def __call__(self, request: Request) -> Any:
            """Handle API requests"""
            path = request.url.path
            
            if path.endswith("/sdapi/v1/txt2img"):
                return await self.handle_txt2img(request)
            elif path.endswith("/sdapi/v1/img2img"):
                return await self.handle_img2img(request)
            elif path.endswith("/sdapi/v1/options"):
                # Return some default options
                return JSONResponse({
                    "sd_model_checkpoint": "kandinsky",
                    "sd_vae": "Automatic",
                })
            elif path.endswith("/sdapi/v1/samplers"):
                # Return list of samplers (Kandinsky uses DPM++ 2M Karras)
                return JSONResponse([
                    {"name": "DPM++ 2M Karras", "aliases": ["karras"], "options": {}}
                ])
            else:
                return JSONResponse({"error": f"Unknown endpoint: {path}"}, status_code=404)
        
        async def handle_txt2img(self, request: Request) -> JSONResponse:
            """Convert Automatic1111 txt2img → Kandinsky generate"""
            try:
                # Get Automatic1111 format request
                auto1111_data = await request.json()
                
                # Transform to Kandinsky format
                kandinsky_data = {
                    "prompt": auto1111_data.get("prompt", ""),
                    "negative_prompt": auto1111_data.get("negative_prompt"),
                    "num_inference_steps": auto1111_data.get("steps", 50),  # steps → num_inference_steps
                    "guidance_scale": auto1111_data.get("cfg_scale", 7.5),  # cfg_scale → guidance_scale
                    "width": auto1111_data.get("width", 512),
                    "height": auto1111_data.get("height", 512),
                    "seed": auto1111_data.get("seed"),
                }
                # Remove None values
                kandinsky_data = {k: v for k, v in kandinsky_data.items() if v is not None}
                
                # Call existing Kandinsky endpoint
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.kandinsky_url}/generate",
                        json=kandinsky_data
                    )
                    response.raise_for_status()
                    kandinsky_result = response.json()
                
                # Transform response back to Automatic1111 format
                # Kandinsky returns: {"image": "data:image/png;base64,..."}
                # Automatic1111 expects: {"images": ["base64"], "info": "..."}
                
                image_data_uri = kandinsky_result.get("image", "")
                # Extract base64 from data URI
                if "," in image_data_uri:
                    image_b64 = image_data_uri.split(",", 1)[1]
                else:
                    image_b64 = image_data_uri
                
                # Build Automatic1111 response
                seed = auto1111_data.get("seed", -1)
                return JSONResponse({
                    "images": [image_b64],  # Array of base64 strings
                    "parameters": {
                        "prompt": auto1111_data.get("prompt", ""),
                        "negative_prompt": auto1111_data.get("negative_prompt", ""),
                        "steps": auto1111_data.get("steps", 50),
                        "cfg_scale": auto1111_data.get("cfg_scale", 7.5),
                        "width": auto1111_data.get("width", 512),
                        "height": auto1111_data.get("height", 512),
                        "seed": seed,
                    },
                    "info": json.dumps({
                        "prompt": auto1111_data.get("prompt", ""),
                        "all_prompts": [auto1111_data.get("prompt", "")],
                        "negative_prompt": auto1111_data.get("negative_prompt", ""),
                        "seed": seed,
                        "all_seeds": [seed],
                        "subseed": -1,
                        "all_subseeds": [-1],
                        "subseed_strength": 0,
                        "width": auto1111_data.get("width", 512),
                        "height": auto1111_data.get("height", 512),
                        "sampler_index": 0,
                        "sampler": "DPM++ 2M Karras",
                        "cfg_scale": auto1111_data.get("cfg_scale", 7.5),
                        "steps": auto1111_data.get("steps", 50),
                    })
                })
            except httpx.HTTPStatusError as e:
                return JSONResponse(
                    {"error": f"Kandinsky endpoint error: {e.response.text}"},
                    status_code=e.response.status_code
                )
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
        
        async def handle_img2img(self, request: Request) -> JSONResponse:
            """Convert Automatic1111 img2img → Kandinsky image-to-image"""
            try:
                # Get Automatic1111 format request
                auto1111_data = await request.json()
                
                # Automatic1111 uses "init_images" array, we need first image
                if "init_images" not in auto1111_data or len(auto1111_data["init_images"]) == 0:
                    return JSONResponse({"error": "init_images required"}, status_code=400)
                
                # Transform to Kandinsky format
                # Automatic1111 sends base64 strings, Kandinsky expects data URI
                init_image_b64 = auto1111_data["init_images"][0]
                if not init_image_b64.startswith("data:"):
                    image_data_uri = f"data:image/png;base64,{init_image_b64}"
                else:
                    image_data_uri = init_image_b64
                
                kandinsky_data = {
                    "prompt": auto1111_data.get("prompt", ""),
                    "negative_prompt": auto1111_data.get("negative_prompt"),
                    "image": image_data_uri,
                    "strength": auto1111_data.get("denoising_strength", 0.75),  # denoising_strength → strength
                    "num_inference_steps": auto1111_data.get("steps", 50),
                    "guidance_scale": auto1111_data.get("cfg_scale", 7.5),
                    "seed": auto1111_data.get("seed"),
                }
                # Remove None values
                kandinsky_data = {k: v for k, v in kandinsky_data.items() if v is not None}
                
                # Call existing Kandinsky endpoint
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{self.kandinsky_url}/image-to-image",
                        json=kandinsky_data
                    )
                    response.raise_for_status()
                    kandinsky_result = response.json()
                
                # Transform response back to Automatic1111 format
                image_data_uri = kandinsky_result.get("image", "")
                if "," in image_data_uri:
                    image_b64 = image_data_uri.split(",", 1)[1]
                else:
                    image_b64 = image_data_uri
                
                seed = auto1111_data.get("seed", -1)
                return JSONResponse({
                    "images": [image_b64],
                    "parameters": {
                        "prompt": auto1111_data.get("prompt", ""),
                        "negative_prompt": auto1111_data.get("negative_prompt", ""),
                        "init_images": ["<base64_image>"],  # Don't echo back full image
                        "denoising_strength": auto1111_data.get("denoising_strength", 0.75),
                        "steps": auto1111_data.get("steps", 50),
                        "cfg_scale": auto1111_data.get("cfg_scale", 7.5),
                        "width": auto1111_data.get("width", 512),
                        "height": auto1111_data.get("height", 512),
                        "seed": seed,
                    },
                    "info": json.dumps({
                        "prompt": auto1111_data.get("prompt", ""),
                        "all_prompts": [auto1111_data.get("prompt", "")],
                        "negative_prompt": auto1111_data.get("negative_prompt", ""),
                        "seed": seed,
                        "all_seeds": [seed],
                        "denoising_strength": auto1111_data.get("denoising_strength", 0.75),
                        "width": auto1111_data.get("width", 512),
                        "height": auto1111_data.get("height", 512),
                        "sampler_index": 0,
                        "sampler": "DPM++ 2M Karras",
                        "cfg_scale": auto1111_data.get("cfg_scale", 7.5),
                        "steps": auto1111_data.get("steps", 50),
                    })
                })
            except httpx.HTTPStatusError as e:
                return JSONResponse(
                    {"error": f"Kandinsky endpoint error: {e.response.text}"},
                    status_code=e.response.status_code
                )
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )
    
    return Automatic1111Adapter


def create_app(kandinsky_base_url: str):
    """Create the adapter app"""
    deployment_class = create_deployment(kandinsky_base_url)
    return deployment_class.bind(kandinsky_base_url=kandinsky_base_url)
