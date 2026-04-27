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

echo "Starting NeonBeam Lens on port $VISION_PORT (USE_HAILO=$USE_HAILO)..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$VISION_PORT"
