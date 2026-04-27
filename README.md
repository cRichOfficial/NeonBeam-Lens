# NeonBeam Lens
### Machine Vision Layer — AI Workpiece Detection & Camera Calibration

NeonBeam Lens is a Python FastAPI service designed for the **Raspberry Pi 5** (with **Hailo-8L AI NPU**). It provides the "eyes" for the NeonBeam suite, handling real-time workpiece detection, camera-to-bed calibration, and automatic design alignment.

---

## Features

- **AprilTag Calibration**: Automatic mapping of camera pixels to physical laser coordinates (mm) using standard AprilTags.
- **NPU-Accelerated Detection**: High-speed workpiece boundary detection using the Hailo-8L NPU.
- **Auto-Layout Engine**: Calculates optimal rotation, scale, and placement for artwork based on detected materials.
- **Live Preview**: Real-time MJPEG stream for remote visual alignment.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VISION_PORT` | `8001` | API listening port. |
| `VISION_CAMERA_ID` | `0` | OpenCV camera index. |
| `USE_HAILO` | `False` | Enable Hailo-8L NPU acceleration. |
| `VISION_DEBUG_LOGGING` | `True` | Enable verbose service logs. |

> **Note**: If `USE_HAILO=True`, the service will fail to start if Hailo hardware or the `hailort` library is missing.

---

## API Endpoints

### 1. AprilTag Generation & Calibration
- **`GET /api/apriltag/generate/{tag_id}`**: Returns a printable AprilTag (Tag36h11).
- **`POST /api/lens/calibrate`**: Calibrates the camera using detected tags and physical mm coordinates.
  ```json
  {
    "tags": [{"id": 1, "physical_x": 0, "physical_y": 0, "size_mm": 50}]
  }
  ```

### 2. Detection & Vision
- **`GET /api/lens/detect`**: Returns a list of detected workpieces with their physical boundaries (`corners_mm`).
- **`GET /api/lens/stream`**: Real-time MJPEG preview stream.
- **`GET /api/lens/frame`**: Captures a single JPEG frame.

### 3. Transformation (Tap-to-Place)
- **`POST /api/lens/transform`**: Calculates alignment parameters.
  - **Inputs**: `workpiece_id`, `design_width_mm`/`design_height_mm` OR `design_file` + `dpi`.
  - **Returns**: Rotation (deg), Scale, and Translation (mm).

---

## Development

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### Running Locally (CPU Fallback)
```bash
export USE_HAILO=False
uvicorn app.main:app --port 8001 --reload
```

---

## Production (Raspberry Pi 5)

### Build & Run (Docker)
```bash
# Ensure Hailo drivers are installed on the host
docker build -t neonbeam-lens .
docker run -d \
  --name neonbeam-lens \
  --device /dev/video0:/dev/video0 \
  --device /dev/hailo0:/dev/hailo0 \
  -e USE_HAILO=True \
  -p 8001:8001 \
  neonbeam-lens
```

---

## Project Layout

```
app/
├── main.py             # FastAPI entry point & routes
├── services/
│   ├── camera.py       # Frame acquisition & streaming
│   ├── calibration.py  # AprilTag & Homography logic
│   ├── inference.py    # NPU Inference & workpiece detection
│   └── transform.py    # Auto-layout calculation engine
├── data/               # Persistent calibration data
└── models/             # Hailo .hef model files
```

> Large model files (`.hef`) should be stored in Git LFS or a separate artifact registry rather than committed directly.
