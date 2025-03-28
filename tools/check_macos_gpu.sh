#!/bin/bash

# Script to check GPU usage on macOS
# Uses various macOS-specific tools

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script only works on macOS"
    exit 1
fi

echo "=== macOS System Information ==="
uname -a
system_profiler SPHardwareDataType | grep -i chip

echo "=== GPU Monitoring on macOS ==="

# Check if Metal is supported
system_profiler SPDisplaysDataType | grep -i metal

# Use powermetrics to check GPU utilization
echo "Running 3-second GPU utilization sample with powermetrics..."
echo "(This requires sudo access)"
sudo powermetrics --samplers gpu_power -n 1 -i 3000 2>/dev/null | grep -i "gpu\|power\|frequency\|util"

echo ""
echo "=== Additional Monitoring Options ==="
echo "1. Open Activity Monitor to view GPU usage graphically:"
echo "   ./tools/open_activity_monitor.sh"
echo ""
echo "2. For live GPU monitoring during script execution, run in another terminal:"
echo "   sudo powermetrics --samplers gpu_power -i 1000"
echo ""
echo "3. Check PyTorch GPU support:"
echo "   ./tools/check_gpu.py" 