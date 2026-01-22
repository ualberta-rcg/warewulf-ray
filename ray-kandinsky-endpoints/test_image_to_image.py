#!/usr/bin/env python3
"""
Test script for Kandinsky image-to-image endpoint
"""

import base64
import requests
from PIL import Image
import io
import sys
import json
import argparse

def prepare_image(image_path, target_size=(768, 768)):
    """Load and prepare image for API"""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_b64}"

def test_image_to_image(
    base_url,
    image_path,
    prompt,
    output_path=None,
    strength=0.75,
    num_inference_steps=75,
    width=768,
    height=768
):
    """Test image-to-image endpoint"""
    
    print(f"🖼️  Loading image: {image_path}")
    try:
        image_data = prepare_image(image_path, target_size=(width, height))
        print(f"✅ Image prepared: {width}x{height}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return False
    
    # Build payload
    data = {
        "prompt": prompt,
        "image": image_data,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
    }
    
    endpoint = f"{base_url}/image-to-image"
    print(f"\n📤 Sending request to: {endpoint}")
    print(f"   Prompt: {prompt}")
    print(f"   Strength: {strength}")
    print(f"   Steps: {num_inference_steps}")
    
    try:
        resp = requests.post(
            endpoint,
            json=data,
            stream=True,   # Important: do not buffer response
            timeout=None   # Allow long inference
        )
        
        print(f"\n📥 Response status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"❌ Request failed with status {resp.status_code}")
            try:
                error_data = resp.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
                if 'traceback' in error_data:
                    print(f"\n   Traceback:\n{error_data['traceback']}")
            except:
                print(f"   Response: {resp.text[:500]}")
            return False
        
        # Save response
        if output_path:
            print(f"💾 Saving output to: {output_path}")
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Output saved to {output_path}")
        else:
            # Try to parse JSON and extract image
            try:
                result = resp.json()
                if "image" in result:
                    # Extract base64 image
                    image_b64 = result["image"].split(",")[1] if "," in result["image"] else result["image"]
                    image_bytes = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save as PNG
                    output_path = "output_image.png"
                    image.save(output_path)
                    print(f"✅ Image saved to {output_path}")
                    
                    # Print metrics if available
                    if "compute_metrics" in result:
                        metrics = result["compute_metrics"]
                        print(f"\n📊 Metrics:")
                        print(f"   Duration: {metrics.get('query_duration_seconds', 'N/A')}s")
                        print(f"   GPU Memory: {metrics.get('replica_gpu_memory_allocated_mb', 'N/A')} MB")
                        print(f"   GPU Utilization: {metrics.get('replica_gpu_utilization_percent', 'N/A')}%")
                else:
                    print(f"⚠️  Response doesn't contain 'image' field")
                    print(f"   Response keys: {list(result.keys())}")
            except Exception as parse_err:
                print(f"⚠️  Could not parse JSON response: {parse_err}")
                print(f"   Saving raw response to output.raw")
                with open("output.raw", "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Kandinsky image-to-image endpoint")
    parser.add_argument(
        "--url",
        type=str,
        default="http://172.26.92.232:9201/kandinsky-kandinsky3-1/v1",
        help="Base URL for the API (default: http://172.26.92.232:9201/kandinsky-kandinsky3-1/v1)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="artistic face painting with wild, crazy colorful hair, whimsical and fun, highly detailed",
        help="Text prompt for image transformation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: output_image.png or output.raw)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Transformation strength (0.0-1.0, default: 0.75)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=75,
        help="Number of inference steps (default: 75)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Image width (default: 768)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height (default: 768)"
    )
    
    args = parser.parse_args()
    
    success = test_image_to_image(
        base_url=args.url,
        image_path=args.image,
        prompt=args.prompt,
        output_path=args.output,
        strength=args.strength,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
