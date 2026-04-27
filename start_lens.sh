#!/bin/bash
# NeonBeam Lens — Start Script

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set default environment variables if not already set
export USE_HAILO=${USE_HAILO:-True}
export VISION_CAMERA_ID=${VISION_CAMERA_ID:-0}
export VISION_PORT=${VISION_PORT:-8001}

# Raspberry Pi 5 libcamera compatibility layer
# This allows OpenCV to access the CSI camera via V4L2
V4L2_WRAPPER="/usr/libexec/aarch64-linux-gnu/libcamera/v4l2-compat.so"
if [ ! -f "$V4L2_WRAPPER" ]; then
    V4L2_WRAPPER="/usr/lib/aarch64-linux-gnu/libcamerav4l2.so"
fi

if [ -f "$V4L2_WRAPPER" ]; then
    echo "Found Pi 5 V4L2 wrapper at $V4L2_WRAPPER, preloading..."
    export LD_PRELOAD="$V4L2_WRAPPER"
fi

echo "Starting NeonBeam Lens on port $VISION_PORT (USE_HAILO=$USE_HAILO)..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$VISION_PORT"
