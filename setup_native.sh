#!/bin/bash
# NeonBeam Lens — Native Setup Script for Raspberry Pi 5
# Sets up a Python virtual environment and installs all dependencies.
# Detection is performed entirely in software (OpenCV + NumPy).
# No NPU / Hailo hardware is required.

set -e

echo "--- NeonBeam Lens Native Setup ---"

# 1. Update system and install dependencies
echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    python3-libcamera \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libv4l-dev \
    v4l-utils

# 2. Create virtual environment
echo "[2/4] Creating virtual environment (.venv) with system package access..."
# We use --system-site-packages so the venv can access picamera2 and libcamera
# bindings which are typically managed by the system package manager on the Pi.

if [ -d ".venv" ]; then
    rm -rf .venv
fi

python3 -m venv .venv --system-site-packages
echo "Created .venv."

# 3. Install Python dependencies
echo "[3/4] Installing Python requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade typing_extensions
pip install -r requirements.txt

# 4. Create necessary directories
echo "[4/4] Creating data directories..."
mkdir -p calibration_data

# Make start script executable
chmod +x start_lens.sh

# 5. Setup systemd service
echo "Registering systemd service..."
SERVICE_PATH="/etc/systemd/system/neonbeam-lens.service"
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=NeonBeam Lens Vision Service
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
echo "Copy .env.example to .env and edit your settings:"
echo "  cp .env.example .env"
echo ""
echo "To start the service:"
echo "  sudo systemctl start neonbeam-lens"
echo ""
echo "To view logs:"
echo "  journalctl -u neonbeam-lens -f"
echo "------------------------------------------------"
