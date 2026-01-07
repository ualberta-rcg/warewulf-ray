"""
Example client for Stable Diffusion and Kandinsky3 endpoints
"""
import requests
import time
from typing import Optional, Dict

class ModelClient:
    def __init__(self, base_url: str = "http://localhost:8000", endpoint: str = ""):
        self.base_url = base_url
        self.endpoint = f"{base_url}{endpoint}"
    
    def wait_for_ready(self, max_wait: int = 300, poll_interval: float = 1.0) -> bool:
        """Poll until model is ready (handles scale-to-zero)"""
        start_time = time.time()
        health_url = f"{self.endpoint}/health"
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ready":
                        print(f"✓ Model is ready! (waited {int(time.time() - start_time)}s)")
                        return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"⏳ Waiting for model... ({elapsed}s)")
        
        print(f"✗ Model failed to become ready within {max_wait}s")
        return False
    
    def generate(self, prompt: str, **kwargs) -> Optional[Dict]:
        """Generate image with automatic polling"""
        if not self.wait_for_ready():
            return None
        
        data = {
            "prompt": prompt,
            "request_id": f"req_{int(time.time() * 1000)}",
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/generate",
                json=data,
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None
    
    def get_metrics(self) -> Optional[Dict]:
        """Get usage metrics summary"""
        try:
            response = requests.get(f"{self.endpoint}/metrics/summary", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting metrics: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Stable Diffusion client
    sd_client = ModelClient(endpoint="/api/v1/stable-diffusion")
    
    print("Generating with Stable Diffusion...")
    result = sd_client.generate(
        "a beautiful landscape with mountains",
        steps=25,
        width=512,
        height=512
    )
    if result:
        print(f"✓ Generated! Latency: {result['usage_stats']['latency_ms']}ms")
    
    # Kandinsky3 client
    kandinsky_client = ModelClient(endpoint="/api/v1/kandinsky3")
    
    print("\nGenerating with Kandinsky3...")
    result = kandinsky_client.generate(
        "futuristic city at sunset",
        steps=30,
        width=1024,
        height=1024
    )
    if result:
        print(f"✓ Generated! Latency: {result['usage_stats']['latency_ms']}ms")
    
    # Get metrics
    print("\nStable Diffusion metrics:")
    print(sd_client.get_metrics())
    
    print("\nKandinsky3 metrics:")
    print(kandinsky_client.get_metrics())
