#!/bin/bash
# Check disk space and suggest cleanup

echo "💾 Checking Disk Space"
echo "===================="
echo ""

# Check overall disk usage
echo "📊 Overall Disk Usage:"
df -h | grep -E "Filesystem|/dev/|/data"
echo ""

# Check specific directories that might be large
echo "📁 Large Directories:"
echo "   /opt/ray:"
du -sh /opt/ray 2>/dev/null || echo "   (not accessible)"
echo "   /data:"
du -sh /data 2>/dev/null || echo "   (not accessible)"
echo "   /tmp:"
du -sh /tmp 2>/dev/null || echo "   (not accessible)"
echo ""

# Check pip cache
echo "📦 Pip Cache:"
PIP_CACHE=$(python3 -m pip cache dir 2>/dev/null || echo "/root/.cache/pip")
if [ -d "$PIP_CACHE" ]; then
    du -sh "$PIP_CACHE" 2>/dev/null
    echo "   Clean with: python3 -m pip cache purge"
else
    echo "   (not found)"
fi
echo ""

# Check Ray cache
echo "🔧 Ray Cache:"
RAY_CACHE="${HOME}/.cache/ray"
if [ -d "$RAY_CACHE" ]; then
    du -sh "$RAY_CACHE" 2>/dev/null
    echo "   Clean with: rm -rf $RAY_CACHE"
else
    echo "   (not found)"
fi
echo ""

# Suggestions
echo "💡 Suggestions to free space:"
echo "   1. Clean pip cache: python3 -m pip cache purge"
echo "   2. Clean Ray cache: rm -rf ~/.cache/ray"
echo "   3. Remove old Docker images: docker system prune -a"
echo "   4. Check /tmp for large files: find /tmp -type f -size +100M"
echo "   5. Check for large log files: find /var/log -type f -size +100M"
