# Ray Serve GraphCast Weather Forecasting Endpoints

This directory contains Ray Serve deployment for GraphCast weather forecasting models with autoscaling and dynamic input discovery.

## Features

- ✅ **Autoscaling with scale-to-zero** - Models scale down after 5 minutes of inactivity
- ✅ **Dynamic schema discovery** - Automatically discovers model inputs and outputs
- ✅ **GPU support** - Automatic GPU detection and usage
- ✅ **Compute metrics** - CPU, GPU, and memory usage in API responses
- ✅ **Polling support** - HTTP 202 responses when scaling up from zero

## Quick Start

### Deploy GraphCast Model

```bash
cd ray-graphcast-endpoints

# Deploy default GraphCast model (google-deepmind/graphcast)
./deploy_graphcast.sh

# Or deploy a specific model
./deploy_graphcast.sh "google-deepmind/graphcast" "my-graphcast-app"
```

### Using Python directly

```bash
/opt/ray/bin/python deploy_graphcast_v2.py \
  --model "google-deepmind/graphcast" \
  --app-name "graphcast-weather"
```

## API Endpoints

Once deployed, the following endpoints are available:

- **Schema Discovery**: `GET /{app-name}/v1/schema` - Discover model inputs and capabilities
- **Health Check**: `GET /{app-name}/v1/health` - Check if model is ready
- **List Models**: `GET /{app-name}/v1/models` - List available models
- **Forecast**: `POST /{app-name}/v1/forecast` - Generate weather forecast

## Example Usage

### Get Schema

```bash
curl http://<head-node-ip>:8000/graphcast-google-deepmind-graphcast/v1/schema
```

### Generate Forecast

```bash
curl -X POST http://<head-node-ip>:8000/graphcast-google-deepmind-graphcast/v1/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "forecast_hours": 6
  }'
```

## Model Loading

GraphCast models are loaded from HuggingFace automatically. The model will:
1. Download and cache model weights (if not already cached)
2. Load on GPU (if available)
3. Discover input/output schema automatically
4. Scale to zero after 5 minutes of inactivity

## Notes

- GraphCast models may require specific input formats (atmospheric data, coordinates, etc.)
- The actual input/output format depends on the specific GraphCast model version
- Check the `/schema` endpoint to see what inputs the model expects
- Models are cached in `/data/models` (NFS shared storage)

## Troubleshooting

- **Model not loading**: Check logs for errors, ensure HuggingFace token is set if needed
- **GPU not available**: Model will fall back to CPU (slower)
- **Scale-to-zero**: First request may take 60+ seconds while model loads
