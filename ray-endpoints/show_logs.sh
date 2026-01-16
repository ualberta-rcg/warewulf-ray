#!/bin/bash
# Quick script to show where Ray logs are and how to view them

echo "📋 Ray Log Locations"
echo "==================="
echo ""

# Find Ray session directories
RAY_SESSION_DIRS=$(find /tmp/ray -maxdepth 1 -type d -name "session_*" 2>/dev/null | head -1)

if [ -z "$RAY_SESSION_DIRS" ]; then
    echo "⚠ No Ray session directories found in /tmp/ray"
    echo "  Ray may not be running or logs are elsewhere"
    echo ""
    echo "Try:"
    echo "  ls -la /tmp/ray/"
    echo "  ls -la /var/log/ray/"
    exit 0
fi

echo "Session directory: $RAY_SESSION_DIRS"
echo ""

# Show log structure
if [ -d "$RAY_SESSION_DIRS/logs" ]; then
    echo "📁 Log directory structure:"
    ls -lh "$RAY_SESSION_DIRS/logs" 2>/dev/null | head -20
    echo ""
    
    # Show serve logs specifically
    if [ -d "$RAY_SESSION_DIRS/logs/serve" ]; then
        echo "📁 Serve logs:"
        ls -lh "$RAY_SESSION_DIRS/logs/serve" 2>/dev/null | head -20
        echo ""
    fi
    
    # Show replica logs
    echo "📁 Replica/Worker logs:"
    find "$RAY_SESSION_DIRS/logs" -name "*replica*" -o -name "*worker*" -o -name "*serve*" 2>/dev/null | head -10
    echo ""
fi

echo "🔍 View logs with:"
echo ""
echo "  # View all logs in session directory"
echo "  tail -f $RAY_SESSION_DIRS/logs/*.log"
echo ""
echo "  # View serve logs"
echo "  tail -f $RAY_SESSION_DIRS/logs/serve/*.log"
echo ""
echo "  # View specific deployment logs (replace DEPLOYMENT_NAME)"
echo "  find $RAY_SESSION_DIRS/logs -name '*DEPLOYMENT_NAME*' -exec tail -f {} +"
echo ""
echo "  # Use Ray CLI to view logs"
echo "  /opt/ray/bin/ray logs"
echo ""
echo "  # View logs via Ray dashboard"
echo "  http://\$(hostname -I | awk '{print \$1}'):8265"
echo ""
echo "  # View systemd logs (if using systemd)"
echo "  sudo journalctl -u ray-head.service -f"
echo "  sudo journalctl -u ray-worker.service -f"
echo ""
