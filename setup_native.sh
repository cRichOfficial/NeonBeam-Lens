#!/bin/bash
# NeonBeam Lens — Native Setup Script for Raspberry Pi 5
# This script sets up a Python virtual environment and installs dependencies.

set -e

echo "--- NeonBeam Lens Native Setup ---"

# 1. Update system and install dependencies
echo "[1/5] Installing basic system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libv4l-dev \
    v4l-utils \
    python3-hailort

# Auto-detect which Python version Hailo is actually compiled for
echo "Detecting Hailo Python binding version..."
HAILO_SO=$(find /usr/lib/python3/dist-packages /usr/lib/python3.* -name "_hailort*.so" 2>/dev/null | head -n 1)

if [ -z "$HAILO_SO" ]; then
    echo "WARNING: Could not find compiled Hailo Python bindings. Using default python3."
    TARGET_PY="python3"
else
    # Extract version (e.g. 311 from _hailort.cpython-311-aarch64-linux-gnu.so)
    VER_STR=$(echo "$HAILO_SO" | grep -oP 'cpython-\K\d+')
    if [ -n "$VER_STR" ]; then
        TARGET_PY="python${VER_STR:0:1}.${VER_STR:1}"
        echo "Found Hailo bindings for $TARGET_PY ($HAILO_SO)"
    else
        TARGET_PY="python3"
    fi
fi

echo "Installing $TARGET_PY virtual environment packages..."
sudo apt-get install -y $TARGET_PY ${TARGET_PY}-venv ${TARGET_PY}-dev

# 2. Create virtual environment
echo "[2/5] Creating virtual environment (.venv) using $TARGET_PY..."

if [ -d ".venv" ]; then
    rm -rf .venv
fi

$TARGET_PY -m venv .venv --system-site-packages
echo "Created .venv with $TARGET_PY and --system-site-packages."

# 3. Install Python dependencies
echo "[3/5] Installing Python requirements..."
source .venv/bin/activate
pip install --upgrade pip

# Upgrade typing_extensions first to bypass the broken Debian package issue
pip install --upgrade typing_extensions

# Install the rest normally so it can inherit pre-compiled system packages like numpy
pip install -r requirements.txt

# 4. Create necessary directories
echo "[4/5] Creating data directories..."
mkdir -p calibration_data app/models

# 5. Symlink the Hailo segmentation model installed by hailo-all
echo "[5/5] Linking Hailo segmentation model..."
HEF_PATH="/usr/share/hailo-models/yolov8s_seg.hef"
LOCAL_HEF="app/models/yolov8s_seg.hef"

if [ -f "$HEF_PATH" ]; then
    if [ ! -e "$LOCAL_HEF" ]; then
        ln -s "$HEF_PATH" "$LOCAL_HEF"
        echo "  Symlinked $HEF_PATH -> $LOCAL_HEF"
    else
        echo "  $LOCAL_HEF already exists, skipping."
    fi
else
    echo "  $HEF_PATH not found. The hailo-models package does not include the segmentation model."
    echo "  Downloading yolov8s_seg.hef directly from Hailo Model Zoo..."
    wget -qO "$LOCAL_HEF" "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l/yolov8s_seg.hef"
    echo "  Downloaded successfully to $LOCAL_HEF"
fi

# Make start script executable
chmod +x start_lens.sh

# 6. Setup systemd service
echo "[5/5] Registering systemd service..."
SERVICE_PATH="/etc/systemd/system/neonbeam-lens.service"
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=NeonBeam Lens Vision AI Service
After=network.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
ExecStart=$CURRENT_DIR/start_lens.sh
Restart=always
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-$CURRENT_DIR/.env

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable neonbeam-lens

echo "------------------------------------------------"
echo "Setup Complete!"
echo ""
echo "The service has been registered and enabled."
echo "To start it now:"
echo "  sudo systemctl start neonbeam-lens"
echo ""
echo "To view logs:"
echo "  journalctl -u neonbeam-lens -f"
echo "------------------------------------------------"
