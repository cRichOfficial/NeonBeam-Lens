# NeonBeam Lens
### Machine Vision Layer — AI Workpiece Detection & Camera Calibration

NeonBeam Lens is the Python FastAPI service that runs on a **Raspberry Pi 5** equipped with a **Hailo-8L AI NPU Accelerator**. It processes overhead camera feeds to detect workpiece boundaries, perform visual calibration, and provide real-time alignment overlays inside NeonBeam OS's Design Studio.

> **Part of the NeonBeam Suite.** See the [root README](../README.md) for the full system architecture.

---

## Stack

| Layer | Technology |
|---|---|
| API Framework | Python 3.11 + FastAPI |
| Computer Vision | OpenCV (`opencv-python`) |
| Numerical | NumPy |
| NPU Inference | HailoRT (`hailort`) + Hailo Python API |
| Hardware Target | Raspberry Pi 5 + Hailo-8L M.2 HAT |

---

## Capabilities (Planned / In Development)

- **Camera Calibration** — Lens undistortion, de-warping, and homography transforms to map camera pixels to real-world machine coordinates.
- **Workpiece Detection** — NPU-accelerated inference to detect the bounding region of irregularly shaped materials on the cutting bed.
- **Artwork Overlay** — Provides a real-time camera frame endpoint (`/api/lens/frame`) so NeonBeam OS can composite the user's artwork directly over a live view of the bed.
- **Auto Orientation** — Detects rotation and skew of rectangular media and automatically adjusts the GCode origin in NeonBeam OS.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VISION_PORT` | `8001` | Port the FastAPI service listens on |
| `VISION_CAMERA_ID` | `0` | OpenCV camera index (0 = first camera) |
| `VISION_DEBUG_LOGGING` | `True` | Enable verbose OpenCV + inference debug output |

---

## Development — Full Stack on One Host

> **Note:** Full NPU functionality requires the Hailo-8L hardware. In development on a standard laptop or desktop, NeonBeam Lens launches with a **software fallback** — camera frames are processed via CPU OpenCV and NPU inference calls are mocked/skipped.

### Prerequisites

- Python 3.11+
- Docker Desktop (optional — native also works)
- A webcam (any USB camera works for development testing)

### Native (Recommended for Initial Development)

```bash
cd machine_vision
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt

export VISION_CAMERA_ID=0
export VISION_DEBUG_LOGGING=True
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The API is available at **http://localhost:8001**.  
Interactive docs: **http://localhost:8001/docs** (Swagger UI).

### Docker (Development)

```bash
# From the repo root:
docker compose up machine-vision

# Tail logs:
docker compose logs -f machine-vision
```

> Camera passthrough inside Docker requires additional device mapping — see Production section.

---

## Production (Raspberry Pi 5 + Hailo-8L)

### Hardware Requirements

- Raspberry Pi 5 (4 GB or 8 GB RAM)
- Hailo-8L M.2 HAT+ installed in the PCIe 2.0 slot
- Raspberry Pi AI Kit drivers installed (HailoRT runtime)
- Overhead USB or CSI camera

### Prerequisites on the Pi

```bash
# 1. Install HailoRT runtime (follow Hailo's official Pi 5 guide)
#    https://hailo.ai/developer-zone/documentation/hailort/
#    After installation, verify:
hailortcli fw-control identify

# 2. Install Docker on Pi 5
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### Steps

```bash
# 1. Clone the NeonBeam Lens repo onto the Pi
git clone <your-remote-url> neonbeam-lens
cd neonbeam-lens

# 2. Configure environment
cp .env.example .env
#    Edit .env — key settings:
#      VISION_CAMERA_ID=0
#      VISION_DEBUG_LOGGING=False   # quieter logs for production

# 3. Build and start
#    The Dockerfile installs OpenCV system dependencies automatically.
docker build -t neonbeam-lens .
docker run -d \
  --name neonbeam-lens \
  --restart unless-stopped \
  --device /dev/video0:/dev/video0 \
  --device /dev/hailo0:/dev/hailo0 \
  -p 8001:8001 \
  --env-file .env \
  neonbeam-lens

# 4. Verify
curl http://localhost:8001/api/lens/health
```

### Camera Device Mapping

| Camera type | Host device | Docker `--device` |
|---|---|---|
| USB webcam | `/dev/video0` | `--device /dev/video0:/dev/video0` |
| Raspberry Pi Camera (libcamera) | `/dev/video0` + media devices | Use `--privileged` or map all `/dev/video*` |
| Hailo-8L NPU | `/dev/hailo0` | `--device /dev/hailo0:/dev/hailo0` |

### Pi 5 Hailo Group Permissions

```bash
# Ensure the user running Docker has access to the Hailo device
sudo usermod -aG hailo $USER
# Log out and back in, then verify:
ls -la /dev/hailo0
```

---

## Git Repository

NeonBeam Lens is designed to be maintained as its own standalone git repository, separate from the other NeonBeam Suite components.

```bash
cd machine_vision
git init
git remote add origin <your-remote-url>
git add .
git commit -m "Initial commit — NeonBeam Lens"
git push -u origin main
```

---

## Project Layout

```
machine_vision/
├── app/
│   ├── main.py             # FastAPI app + route definitions
│   └── services/
│       ├── camera.py       # OpenCV camera management
│       ├── calibration.py  # Homography + undistortion pipeline
│       └── inference.py    # HailoRT NPU inference wrapper
├── models/                 # Compiled Hailo .hef model files (not committed — too large)
├── calibration_data/       # Saved camera calibration matrices
├── requirements.txt
└── Dockerfile
```

> Large model files (`.hef`) should be stored in Git LFS or a separate artifact registry rather than committed directly.
