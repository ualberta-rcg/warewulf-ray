#!/bin/bash
# Quick script to check Ray Serve status and restart if needed

set -e

RAY_PYTHON="/opt/ray/bin/python"

echo "🔍 Checking Ray Serve status..."
"$RAY_PYTHON" -c "
import ray
from ray import serve

ray.init(address='auto', ignore_reinit_error=True)

try:
    status = serve.status()
    print('✅ Ray Serve is running')
    print(f'   Applications: {len(status.applications) if hasattr(status, \"applications\") else \"N/A\"}')
    
    # Check if port 8001 is listening
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('0.0.0.0', 8001))
    sock.close()
    
    if result == 0:
        print('✅ Port 8001 is accessible')
    else:
        print('⚠️  Port 8001 is NOT accessible - Ray Serve may be on wrong port')
        print('   Run: bash restart_serve.sh')
except Exception as e:
    print(f'⚠️  Ray Serve status check failed: {e}')
    print('   Run: bash restart_serve.sh')
"
