#!/bin/bash
# NeonBeam Lens — Native Setup Script for Raspberry Pi 5
# This script sets up a Python virtual environment and installs dependencies.

set -e

echo "--- NeonBeam Lens Native Setup ---"

# 1. Update system and install dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libv4l-dev \
    v4l-utils \
    python3-hailort

# 2. Create virtual environment
echo "[2/4] Creating virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
    echo "Created .venv (using system-site-packages for Hailo access)."
else
    echo ".venv already exists."
fi

# 3. Install Python dependencies
echo "[3/4] Installing Python requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create necessary directories
echo "[4/5] Creating data directories..."
mkdir -p calibration_data app/models

# 5. Symlink the Hailo segmentation model installed by hailo-all
echo "[5/5] Linking Hailo segmentation model..."
HEF_PATH="/usr/share/hailo-models/yolov8s_seg.hef"

if [ ! -f "$HEF_PATH" ]; then
    echo ""
    echo "  WARNING: $HEF_PATH not found."
    echo "  The yolov8s_seg model should be installed as part of 'hailo-all'."
    echo "  Ensure you have run: sudo apt install hailo-all"
    echo "  Then re-run this script."
    echo ""
else
    if [ ! -e "app/models/yolov8s_seg.hef" ]; then
        ln -s "$HEF_PATH" app/models/yolov8s_seg.hef
        echo "  Symlinked $HEF_PATH -> app/models/yolov8s_seg.hef"
    else
        echo "  app/models/yolov8s_seg.hef already linked, skipping."
    fi
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
