# Ray Endpoints

Modular endpoint system for serving models with Ray Serve. Each endpoint is a separate file that can be dynamically discovered and deployed.

## Structure

```
ray-endpoints/
├── __init__.py
├── base.py              # Base endpoint class
├── loader.py            # Endpoint discovery
├── deploy.py            # Deployment script
├── client_example.py    # Example client code
├── endpoints/
│   ├── __init__.py
│   ├── stable_diffusion.py
│   └── kandinsky3.py
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

### Using the Clients

```python
from ray_endpoints.client_example import ModelClient

# Stable Diffusion
sd_client = ModelClient(endpoint="/api/v1/stable-diffusion")
result = sd_client.generate("a beautiful landscape", steps=25)

# Kandinsky3
kandinsky_client = ModelClient(endpoint="/api/v1/kandinsky3")
result = kandinsky_client.generate("futuristic city", steps=30)
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
