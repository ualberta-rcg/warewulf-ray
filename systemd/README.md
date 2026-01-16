# Ray Systemd Service Files

This directory contains systemd service files for running Ray in a cluster setup.

## Files

- **`ray-head.service`**: Service file for the Ray head node
- **`ray-worker.service`**: Service file for Ray worker nodes

## Installation

### On Head Node

```bash
# Copy service file
sudo cp systemd/ray-head.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable ray-head.service
sudo systemctl start ray-head.service

# Check status
sudo systemctl status ray-head.service
```

### On Worker Nodes

```bash
# Copy service file
sudo cp systemd/ray-worker.service /etc/systemd/system/

# Edit the RAY_SERVER environment variable to point to your head node
# Default is 172.26.92.232:6379 - change this to match your setup
sudo sed -i 's/RAY_SERVER=.*/RAY_SERVER=YOUR_HEAD_NODE_IP:6379/' /etc/systemd/system/ray-worker.service

# Or edit manually:
# sudo vi /etc/systemd/system/ray-worker.service
# Change: Environment=RAY_SERVER=172.26.92.232:6379

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable ray-worker.service
sudo systemctl start ray-worker.service

# Check status
sudo systemctl status ray-worker.service
```

## Configuration

### Head Node Configuration

The head node service includes:
- Dashboard on port 8265 (accessible at `http://head-node-ip:8265`)
- Object manager on port 8076
- Metrics export on port 8080
- Object spilling directory: `/data/ray/spill`
- Object store memory: 2GB (2147483648 bytes)
- Resource tag: `{"head": 1}`

To customize, edit `/etc/systemd/system/ray-head.service` and reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ray-head.service
```

### Worker Node Configuration

The worker node service includes:
- `RAY_SERVER`: Address of the head node (default: `172.26.92.232:6379`)
- `RAY_TMPDIR`: Temporary directory for Ray (default: `/tmp/ray`)

To change the head node address, edit the `Environment=RAY_SERVER=...` line in `/etc/systemd/system/ray-worker.service`.

## Troubleshooting

### Check Ray Status

```bash
# On head node
/opt/ray/bin/ray status

# On worker node
/opt/ray/bin/ray status --address="172.26.92.232:6379"
```

### View Logs

```bash
# Head node logs
sudo journalctl -u ray-head.service -f

# Worker node logs
sudo journalctl -u ray-worker.service -f
```

### Stop Services

```bash
# Stop head node
sudo systemctl stop ray-head.service

# Stop worker node
sudo systemctl stop ray-worker.service
```

### Disable Services

```bash
# Disable head node
sudo systemctl disable ray-head.service

# Disable worker node
sudo systemctl disable ray-worker.service
```

## Notes

- The environment variables are merged into the service files (no separate `/etc/sysconfig/ray-worker` file needed)
- Make sure `/data/ray/spill` directory exists on the head node before starting
- Ensure network connectivity between head and worker nodes
- The head node IP address in `ray-worker.service` must match your actual head node
