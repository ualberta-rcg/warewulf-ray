# Ray Endpoints

Modular endpoint system for serving models with Ray Serve. Each endpoint is a separate file that can be dynamically discovered and deployed.

## Structure

```
ray-endpoints/
├── __init__.py
├── base.py              # Base endpoint class
├── loader.py            # Endpoint discovery
├── deploy.py            # Deployment script
├── deploy_triton.py     # Triton deployments
├── endpoints/
│   ├── __init__.py
│   ├── stable_diffusion.py
│   ├── kandinsky3.py
│   ├── stable_diffusion_triton.py
│   ├── kandinsky3_triton.py
│   └── triton_inference.py
└── README.md
```

## Usage

### Deploy All Endpoints

```bash
cd /opt/ray-endpoints
/opt/ray/bin/python deploy.py
```

### Deploy Single Endpoint

```bash
/opt/ray/bin/python deploy.py endpoints/stable_diffusion.py
```

### Using the Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/stable-diffusion/health

# Generate image (POST request)
curl -X POST http://localhost:8000/api/v1/stable-diffusion/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape", "steps": 25}'
```

## Features

- **Scale-to-zero**: Models scale down when idle
- **Automatic polling**: Clients wait for models to load
- **Request queueing**: Built-in queue management
- **Usage metrics**: Saved to `/var/log/ray/*-metrics/` as JSONL files
- **Health checks**: `/health` endpoint for monitoring

## Adding New Endpoints

Create a new file in `endpoints/`:

```python
from ..base import BaseEndpoint

class MyModelEndpoint(BaseEndpoint):
    ROUTE_PREFIX = "/api/v1/my-model"
    DEPLOYMENT_NAME = "my-model"
    
    def load_model(self):
        # Load your model
        return {"loaded": True}
    
    def predict(self, data):
        # Your inference logic
        return {"result": "..."}
```

Then redeploy:
```bash
/opt/ray/bin/python deploy.py
```
