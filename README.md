# NeonBeam Lens
### Machine Vision Layer — AI Workpiece Detection & Camera Calibration

NeonBeam Lens is a Python FastAPI service designed for the **Raspberry Pi 5** (with **Hailo-8L AI NPU**). It provides the "eyes" for the NeonBeam suite, handling real-time workpiece detection, camera-to-bed calibration, and automatic design alignment.

---

## Features

- **AprilTag Calibration**: Automatic mapping of camera pixels to physical laser coordinates (mm) using standard AprilTags.
- **NPU-Accelerated Instance Segmentation**: High-speed workpiece boundary detection using YOLOv8-seg on the Hailo-8L NPU. Falls back to OpenCV contour detection when running without Hailo hardware.
- **Auto-Layout Engine**: Calculates optimal rotation, scale, and placement for artwork based on detected materials.
- **Live Preview**: Real-time MJPEG stream for remote visual alignment.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VISION_PORT` | `8001` | API listening port. |
| `VISION_CAMERA_ID` | `0` | OpenCV camera index. |
| `USE_HAILO` | `False` | Enable Hailo-8L NPU acceleration. |
| `MODEL_PATH` | `app/models/yolov8s_seg.hef` | Path to the Hailo segmentation model. |
| `DETECT_CONF_THRESHOLD` | `0.4` | Minimum detection confidence score (0–1). |
| `MIN_WORKPIECE_AREA_MM2` | `500` | Minimum physical area to report as a workpiece (mm²). |
| `MAX_WORKPIECE_AREA_MM2` | `160000` | Maximum physical area (~400×400mm bed). |
| `HONEYCOMB_KERNEL_SIZE` | `25` | Morphological kernel size for OpenCV fallback (px). |
| `VISION_DEBUG_LOGGING` | `True` | Enable verbose service logs. |

> **Note**: If `USE_HAILO=True`, the service will fail to start if Hailo hardware or the `hailort` library is missing, or if the HEF model is not present.

---

## API Endpoints

### 1. AprilTag Generation
- **`GET /api/apriltag/generate/{tag_id}`** — Returns a printable AprilTag as a PDF (includes cut-line border and ID label).
- **`GET /api/apriltag/batch`** — Returns a multi-page PDF of tags densely packed on 8.5×11 paper (or custom size).
  - Params: `start_id`, `count`, `size_mm`, `dpi`, `paper_width_in`, `paper_height_in`

### 2. Calibration
- **`POST /api/lens/calibrate`** — Calibrates the camera using detected tags and physical mm coordinates.
  ```json
  { "tags": [{"id": 1, "physical_x": 0, "physical_y": 0, "size_mm": 50}] }
  ```
- **`GET /api/lens/calibration`** — Returns the current calibration status, stored homography matrix, and tag metadata.
- **`GET /api/lens/calibration/check`** — Live quality check. Detects visible AprilTags in the current frame and returns per-tag reprojection error in mm. Use this to verify calibration accuracy.

### 3. Detection
- **`GET /api/lens/detect`** — Returns detected workpieces with physical boundaries, segmentation polygon, orientation angle, and area.
  ```json
  {
    "id": "wp_000",
    "class": "workpiece",
    "confidence": 0.87,
    "angle_deg": 12.3,
    "corners_mm": [[x,y],[x,y],[x,y],[x,y]],
    "segmentation_mm": [[x,y], ...],
    "area_mm2": 9800.5
  }
  ```
- **`GET /api/lens/stream`** — Real-time MJPEG preview stream.
- **`GET /api/lens/frame`** — Captures a single JPEG frame.

### 4. Transformation (Tap-to-Place)
- **`POST /api/lens/transform`** — Calculates alignment parameters.
  - **Inputs**: `workpiece_id`, `design_width_mm`/`design_height_mm` OR `design_file` + `dpi`.
  - **Returns**: Rotation (deg), Scale, and Translation (mm).

---

## Production (Raspberry Pi 5 — Native)

### Prerequisites

Ensure the following are installed on the Pi before running `setup_native.sh`:
- HailoRT PCIe Driver (`.deb`) — from [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- HailoRT (`.deb`) — from Hailo Developer Zone
- HailoRT Python Bindings (`.whl`) — from Hailo Developer Zone

### Setup

```bash
git clone <this-repo>
cd NeonBeam-Lens
./setup_native.sh
```

`setup_native.sh` will:
1. Install system dependencies
2. Create a Python virtual environment (using `--system-site-packages` for Hailo access)
3. Install Python requirements
4. Create data directories
5. Install `hailo-apps` and symlink the `yolov8s_seg.hef` model into `app/models/`
6. Register and enable the `neonbeam-lens` systemd service

### Start the Service

```bash
sudo systemctl start neonbeam-lens
journalctl -u neonbeam-lens -f
```

---

## Model Setup

NeonBeam Lens uses `yolov8s_seg` (YOLOv8 Small Instance Segmentation) compiled for the Hailo-8L.

The HEF is installed automatically by `sudo apt install hailo-all` to `/usr/share/hailo-models/`. Running `setup_native.sh` creates a symlink at `app/models/yolov8s_seg.hef` so the service can find it via the `MODEL_PATH` env var.

No manual downloads or recompilation are needed. See `app/models/README.md` for details on switching model sizes.

---

## Development (CPU Fallback)

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### Running Locally (without Hailo hardware)
```bash
export USE_HAILO=False
uvicorn app.main:app --port 8001 --reload
```

When `USE_HAILO=False`, the service uses an OpenCV-based contour detector as a fallback. Detection quality on a black honeycomb bed is limited for low-contrast workpieces; this mode is intended for development and API testing.

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
├── models/             # Hailo .hef model files (see models/README.md)
└── data/               # Persistent calibration data
calibration_data/       # Stored homography matrix
```
