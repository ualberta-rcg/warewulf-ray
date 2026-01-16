# Compute Canada and Dependency Management

This document explains how to use Compute Canada modules and add dependencies to Ray Serve deployments.

## ONNX Models

**Yes, ONNX models work great with Triton!** Triton has excellent support for ONNX Runtime backend.

### Setting up ONNX Models in Triton

1. **Model Repository Structure:**
```
/data/ray-triton/model_repository/
└── your_model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

2. **config.pbtxt for ONNX:**
```protobuf
name: "your_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1, 1000 ]
  }
]
```

3. **ONNX Runtime Backend:**
Triton will automatically use ONNX Runtime when it detects `platform: "onnxruntime_onnx"` in the config.

## Adding Dependencies

### Method 1: Using Runtime Environment (Recommended)

The deployment code now supports `runtime_env` which can include:

**Pip packages:**
```python
runtime_env = {
    "pip": ["numpy", "pandas", "onnxruntime", "psutil"],
}
```

**Environment variables:**
```python
runtime_env = {
    "env_vars": {
        "PYTHONPATH": "/path/to/your/code",
        "LD_LIBRARY_PATH": "/path/to/libs",
    },
}
```

**Both:**
```python
runtime_env = {
    "env_vars": {
        "PYTHONPATH": "/path/to/your/code",
    },
    "pip": ["numpy", "pandas"],
}
```

### Method 2: Using Compute Canada Modules

Since Compute Canada modules are available on all workers, you can use them instead of pip:

**In `deploy.py`, modify the `compute_canada_modules` list:**
```python
compute_canada_modules = []
if os.path.exists("/cvmfs/soft.computecanada.ca/config/profile/bash.sh"):
    # Add your modules here
    compute_canada_modules = [
        "python/3.12",
        "cuda/12.0",
        "cudnn/8.9",
        "onnxruntime/1.16.0",  # If available
    ]
```

**In `triton_inference.py`, modify the `compute_canada_modules` list:**
```python
compute_canada_modules = []
if os.path.exists("/cvmfs/soft.computecanada.ca/config/profile/bash.sh"):
    compute_canada_modules = [
        "python/3.12",
        "cuda/12.0",
        "onnxruntime/1.16.0",
    ]
```

### Method 3: Install in Ray Virtual Environment

Install packages globally in the Ray venv (available on all nodes):
```bash
/opt/ray/bin/pip install onnxruntime numpy pandas
```

This makes packages available to all deployments without runtime_env.

## Fixing the "No module named 'base'" Error

The deployment code now automatically:
1. Sets `PYTHONPATH` in `runtime_env` to include the ray-endpoints directory
2. Adds the path to `sys.path` in `__init__` on worker nodes
3. Ensures the `base` module can be imported on all workers

This should fix the `ModuleNotFoundError: No module named 'base'` issue.

## Example: Adding ONNX Runtime Support

**Option 1: Use Compute Canada module (if available):**
```python
# In deploy.py or triton_inference.py
compute_canada_modules = ["onnxruntime/1.16.0"]
```

**Option 2: Install via pip in runtime_env:**
```python
runtime_env = {
    "pip": ["onnxruntime"],
}
```

**Option 3: Install globally:**
```bash
/opt/ray/bin/pip install onnxruntime
```

## Checking Available Compute Canada Modules

```bash
# List available modules
module avail

# List loaded modules
module list

# Search for specific modules
module avail onnx
module avail cuda
```

## Method 4: Shared Virtual Environment on NFS (Recommended for NFS setups)

Since `/data` is an NFS share accessible on all workers, you can create a shared venv:

**1. Create the shared venv:**
```bash
cd ~/warewulf-ray/ray-endpoints
bash setup_shared_venv.sh
```

Or manually:
```bash
python3.12 -m venv /data/ray-endpoints-venv
/data/ray-endpoints-venv/bin/pip install --upgrade pip
/data/ray-endpoints-venv/bin/pip install onnxruntime psutil requests numpy
```

**2. Install packages in the shared venv:**
```bash
/data/ray-endpoints-venv/bin/pip install package_name
```

**3. The deployment code automatically detects and uses it:**
- Checks for `/data/ray-endpoints-venv`
- Adds it to `PATH` in runtime_env
- Adds it to `sys.path` in `__init__` on workers

**Benefits:**
- ✅ Packages installed once, available on all workers
- ✅ No download/install on each worker node
- ✅ Faster deployment startup
- ✅ Consistent environment across all nodes

## Best Practices

1. **For packages already in `/opt/ray`**: Don't add to runtime_env, they're already available
2. **For NFS setups**: Use shared venv in `/data/ray-endpoints-venv` (fastest, most efficient)
3. **For Compute Canada modules**: Use modules if available (faster, no download)
4. **For custom packages**: Use shared venv > Compute Canada modules > runtime_env pip > global install
5. **For ONNX models**: Ensure ONNX Runtime is available (via shared venv, module, or pip)

## Troubleshooting

**If modules don't load:**
- Check that `/cvmfs/soft.computecanada.ca/config/profile/bash.sh` exists
- Verify module names with `module avail`
- Check logs for module loading errors

**If pip packages fail to install:**
- Check network connectivity on worker nodes
- Verify pip is available in the runtime environment
- Consider installing globally in `/opt/ray` instead
