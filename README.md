# warewulf-ray

A Warewulf-compatible container image for deploying Ray with NVIDIA Triton Inference Server support on HPC clusters. This image is designed for the Digital Research Alliance of Canada (DRAC) infrastructure and provides a complete environment for distributed machine learning inference workloads.

## Overview

This container image provides:
- **Ray** (full installation with all extras: default, serve, tune, rllib, data, train, air, gpu, and more)
- **NVIDIA Triton Inference Server** support:
  - `tritonclient[all]`: Client libraries for connecting to Triton servers (HTTP, gRPC)
  - Triton server binary and Python API from NVIDIA's official image (multi-stage build)
  - Python 3.10 compatibility for `import tritonserver` support
  - Fallback to `nvidia-pytriton` binary if needed
- **NVIDIA GPU driver** support (optional, configurable)
- **Systemd** support for service management
- **Ansible** integration for first-boot configuration
- **CUDA toolkit** via CVMFS (provided by DRAC)

## Features

- Full systemd support for service management
- Complete Ray installation with all extras (Serve, Tune, RLlib, Data, Train, AIR, GPU, etc.)
- GPU support with optional NVIDIA driver installation
- Ray Serve integration with Triton Inference Server (standalone binary approach)
- First-boot configuration via Ansible playbooks
- Optimized for HPC cluster environments (Warewulf)
- Minimal Python system packages (uses virtual environment)

## Requirements

- Ubuntu 24.04 base image
- Warewulf 4.x compatible
- NVIDIA GPUs (if GPU support is enabled)
- CUDA toolkit available via CVMFS (DRAC infrastructure)

## Build Arguments

The image supports the following build arguments:

| Argument | Description | Default |
|-----------|-------------|---------|
| `KERNEL_VERSION` | Linux kernel version to install (e.g., `6.8.0-90-generic`) | Required if `KERNEL_INSTALL_ENABLED=true` |
| `KERNEL_INSTALL_ENABLED` | Enable kernel installation | `false` |
| `NVIDIA_INSTALL_ENABLED` | Enable NVIDIA driver installation | `false` |
| `NVIDIA_DRIVER_URL` | URL to NVIDIA driver installer | Required if `NVIDIA_INSTALL_ENABLED=true` |
| `DISABLE_AUTOLOGIN` | Disable root autologin | `false` |
| `FIRSTBOOT_ENABLED` | Enable first-boot Ansible configuration | `false` |

## Building the Image

### Basic Build

```bash
docker build -t warewulf-ray:latest .
```

### Build with Kernel and NVIDIA Driver

```bash
docker build \
  --build-arg KERNEL_INSTALL_ENABLED=true \
  --build-arg KERNEL_VERSION=6.8.0-90-generic \
  --build-arg NVIDIA_INSTALL_ENABLED=true \
  --build-arg NVIDIA_DRIVER_URL=https://us.download.nvidia.com/XFree86/Linux-x86_64/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run \
  -t warewulf-ray:latest .
```

### Build with First-Boot Configuration

```bash
docker build \
  --build-arg FIRSTBOOT_ENABLED=true \
  -t warewulf-ray:latest .
```

## Image Structure

### Users and Groups

- **root** (UID 0): Default root user with password `changeme` (change on first boot)
- **wwuser** (UID 2000): Warewulf user in `wwgroup` (GID 2000), member of `sudo` group
- **dist** (UID 2001): Distributive.network user in `distgroup` (GID 2001)

### Directories

- `/opt/ray`: Ray virtual environment (Python packages)
- `/etc/ray`: Ray configuration directory
- `/var/log/ray`: Ray log directory
- `/var/lib/ray`: Ray state directory
- `/tmp/ray`: Ray temporary files
- `/local/home`: User home directories

### Services

The image includes the following systemd services:
- `ssh.service`: SSH server
- `rsyslog.service`: System logging
- `auditd.service`: Audit daemon
- `firstboot.service`: First-boot configuration (if enabled)

## Usage

### Starting Ray Cluster

```bash
# Start Ray head node
ray start --head --port=6379

# Start Ray worker node
ray start --address=<head-node-ip>:6379
```

### Using Ray Serve with Triton

```python
from ray import serve
from tritonclient import http as httpclient
import tritonclient.http as tritonhttp

@serve.deployment
class TritonModel:
    def __init__(self):
        self.client = tritonhttp.InferenceServerClient(
            url="localhost:8000"
        )
    
    async def __call__(self, request):
        # Your inference logic here
        pass

# Deploy the model
serve.run(TritonModel.bind())
```

### First-Boot Configuration

If `FIRSTBOOT_ENABLED=true`, the image will:
1. Wait for network connectivity
2. Execute Ansible playbooks from `/etc/ansible/playbooks/*.yaml`
3. Remove the firstboot service after successful completion

Place your Ansible playbooks in `/etc/ansible/playbooks/` directory.

## CUDA Support

The CUDA toolkit is provided via CVMFS (Digital Research Alliance of Canada). Ensure that:
- CVMFS is properly mounted
- The appropriate CUDA version is available in the CVMFS repository
- GPU devices are accessible to the container

## NVIDIA Driver Installation

When `NVIDIA_INSTALL_ENABLED=true` and `KERNEL_INSTALL_ENABLED=true`:
- Downloads and installs the NVIDIA driver from the provided URL
- Configures kernel modules (`nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset`)
- Creates necessary device nodes
- Requires kernel headers to be installed

## Python Environment

Ray (with all extras) and Triton are installed in a virtual environment at `/opt/ray` using **Python 3.10** to:
- Match Triton's Python version (enables `import tritonserver` to work)
- Avoid conflicts with system Python packages
- Comply with PEP 668 (Ubuntu 24.04)
- Isolate dependencies from system packages

**Note:** Python 3.10 is used (instead of 3.12) to ensure compatibility with Triton's Python API bindings.

The `ray[all]` installation includes:
- **default**: Core Ray functionality
- **serve**: Ray Serve for model serving (includes starlette, uvicorn, etc.)
- **tune**: Ray Tune for hyperparameter tuning
- **rllib**: Ray RLlib for reinforcement learning
- **data**: Ray Data for data processing
- **train**: Ray Train for distributed training
- **air**: Ray AIR for end-to-end ML workflows
- **gpu**: GPU support and CUDA integration
- And other optional dependencies

The virtual environment is automatically added to `PATH` for all users.

## Testing

A comprehensive test script is included to verify Ray and Triton installation:

```bash
# Run the integration test suite
./test_ray_triton_integration.sh
```

The test script verifies:
- Ray installation and cluster startup
- Ray Serve deployment and endpoints
- Triton client libraries
- PyTriton functionality
- Resource usage and GPU detection

## Troubleshooting

### Ray Not Found

Ensure the virtual environment is activated:
```bash
source /opt/ray/bin/activate
```

Or verify PATH includes the venv:
```bash
echo $PATH | grep /opt/ray
```

### GPU Not Detected

1. Verify NVIDIA driver is installed:
   ```bash
   nvidia-smi
   ```

2. Check CUDA availability:
   ```bash
   ls /cvmfs/soft.computecanada.ca/config/profile/bash.sh
   source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
   module avail cuda
   ```

3. Verify GPU devices:
   ```bash
   ls -la /dev/nvidia*
   ```

### First-Boot Service Failing

Check logs:
```bash
journalctl -u firstboot.service -n 100
```

Verify Ansible playbooks exist:
```bash
ls -la /etc/ansible/playbooks/
```

## Contributing

This project is maintained by the University of Alberta Research Computing Group. Contributions are welcome!

## License

MIT License - See [LICENSE](LICENSE) file for details.

## References

- [Ray Documentation](https://docs.ray.io/)
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Ray Serve with Triton](https://docs.ray.io/en/latest/serve/tutorials/triton.html)
- [Triton Setup Documentation](ray-endpoints/README_TRITON.md) - Detailed guide for standalone Triton binary
- [Warewulf Documentation](https://warewulf.org/)
- [Digital Research Alliance of Canada](https://alliancecan.ca/)

## Support

For issues and questions:
- Open an issue in the repository
- Contact the University of Alberta Research Computing Group
