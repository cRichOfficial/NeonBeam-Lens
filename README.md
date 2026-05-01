# NeonBeam Lens
### Machine Vision Layer — AI Workpiece Detection & Camera Calibration

NeonBeam Lens is a Python FastAPI service designed for the **Raspberry Pi 5** (with optional **Hailo-8L AI NPU**). It provides the "eyes" for the NeonBeam laser engraver suite, handling real-time workpiece detection, camera-to-bed calibration, and automatic design alignment.

---

## Features

- **AprilTag Calibration**: Automatic mapping of camera pixels to physical laser bed coordinates (mm) using standard AprilTags placed at known positions on the bed corners.
- **NPU-Accelerated Instance Segmentation**: High-speed workpiece boundary detection using YOLOv8-seg on the Hailo-8L NPU. Falls back to an OpenCV morphological contour detector when running without Hailo hardware.
- **Auto-Layout Engine**: Calculates optimal rotation, scale, and placement for artwork based on detected material boundaries.
- **Live Preview**: Real-time MJPEG stream for remote visual alignment.

---

## Camera Hardware

NeonBeam Lens is tested and optimised for the **Raspberry Pi Camera Module 3** (IMX708 sensor, 12MP, 102° HFOV fixed focal length). The service uses the `picamera2` library for direct sensor control and is compatible with any Raspberry Pi camera supported by libcamera.

### HDR Mode (IMX708 / Camera v3)

The IMX708 sensor supports hardware-level HDR via the libcamera pipeline. When enabled via `CAMERA_HDR=True`, the service requests **Sensor-level HDR** from Picamera2, which engages on-sensor dual-exposure merging. This significantly improves dynamic range in mixed-light environments (e.g. a dark honeycomb bed next to a bright reflective work surface).

> **Important:** Hardware HDR on the IMX708 limits the maximum capture resolution to approximately **3MP (2304×1296)**. The service automatically caps `CAMERA_RESOLUTION` to this limit when HDR is active — you do not need to set `CAMERA_RESOLUTION` manually when using HDR.

### Why Capture at Full Resolution?

The service captures at the sensor's full native resolution by default (`CAMERA_RESOLUTION=max`). This gives the ArUco/AprilTag detector a larger image to work with, improving tag detection reliability at the edges of the bed where distortion is highest. Object detection is always performed on a downsampled working copy (see `DETECT_MAX_DIM`), so full-resolution capture does not impact detection speed.

---

## Environment Variables

### Camera

| Variable | Default | Description |
|---|---|---|
| `VISION_CAMERA_ID` | `0` | OpenCV camera index (used when Picamera2 is unavailable). |
| `CAMERA_RESOLUTION` | `max` | Capture resolution. `max` uses the sensor's full native resolution. Override with `WIDTHxHEIGHT` (e.g. `1920x1080`). |
| `CAMERA_HDR` | `False` | Enable sensor-level hardware HDR (IMX708/Camera v3 only). Caps resolution to ~3MP (2304×1296). |
| `FISHEYE_LENS` | `False` | Apply fisheye undistortion (requires manual K/D coefficient tuning). Not needed for the IMX708 102° lens. |
| `WORKSPACE_HEIGHT_MM` | `500` | Physical height of the laser bed in mm. Used to flip the Y-axis so (0,0) is bottom-left. Set to `0` to disable Y-flip. |

### NPU / Hailo

| Variable | Default | Description |
|---|---|---|
| `USE_HAILO` | `False` | Enable Hailo-8L NPU inference. If `True`, the service will fail to start if hardware or the HEF model is missing. |
| `MODEL_PATH` | `app/models/yolov8s_seg.hef` | Path to the Hailo Executable Format segmentation model. |
| `DETECT_CONF_THRESHOLD` | `0.4` | Minimum confidence score for an NPU detection to be accepted (0.0–1.0). |

### Detection & Post-Processing

| Variable | Default | Description |
|---|---|---|
| `DETECT_MAX_DIM` | `1280` | Maximum long-side resolution (px) for the **detection processing pipeline**. See note below. |
| `MIN_WORKPIECE_AREA_MM2` | `500` | Minimum physical area (mm²) to report as a workpiece. Filters out debris and noise. |
| `MAX_WORKPIECE_AREA_MM2` | `160000` | Maximum physical area (mm²). Filters out the bed frame itself (~400×400mm = 160,000 mm²). |
| `HONEYCOMB_KERNEL_SIZE` | `25` | Morphological kernel size (px at 720p reference) for bridging honeycomb gaps within a workpiece blob. |
| `APRILTAG_MASK_PADDING` | `30` | Extra pixels (at 720p reference) to expand the AprilTag exclusion zone. Increase if the white paper backing the tags is being detected as a workpiece. |

### Logging

| Variable | Default | Description |
|---|---|---|
| `VISION_DEBUG_LOGGING` | `True` | `True` = DEBUG level (frame timings, scores, matrices). `False` = INFO level (normal operation). |

---

### About `DETECT_MAX_DIM`

The software object detection pipeline uses morphological image processing operations (Top-Hat transforms, Gaussian blur, morphological close). These operations scale with image resolution as **O(width × height × kernel²)**. At 12MP (4056×3040) with typical kernel sizes, this would take several minutes per frame — far too slow for interactive use.

`DETECT_MAX_DIM` solves this by **downsampling the camera frame to a capped working resolution** before any expensive processing. Only the detection pipeline uses this downsampled copy; the original full-resolution frame is preserved. Detected workpiece contours are scaled back to full-resolution coordinates before the calibration homography maps them to physical mm values, so **accuracy is not affected**.

**Choosing a value:**

| `DETECT_MAX_DIM` | Approx. working size | Typical detection time (Pi 5) | Use when… |
|---|---|---|---|
| `640` | 640×480 | ~0.1s | Fastest; good for large workpieces only |
| `1280` *(default)* | 1280×960 | ~0.3s | Best balance for most workpieces |
| `1920` | 1920×1440 | ~0.8s | Needed for very small or intricate items |
| `2560` | 2560×1920 | ~2s | Maximum detail; likely unnecessary |

**You only need to change this if** detection is either too slow (lower the value) or workpiece outlines are too coarse and missing small items (raise the value). For most use cases the default `1280` is correct.

---

## API Endpoints

### 1. AprilTag Generation
- **`GET /api/apriltag/generate/{tag_id}`** — Returns a printable AprilTag as a PDF (includes cut-line border and ID label).
- **`GET /api/apriltag/batch`** — Returns a multi-page PDF of tags densely packed for printing.
  - Params: `start_id`, `count`, `size_mm`, `dpi`, `paper_width_in`, `paper_height_in`

### 2. Calibration
- **`POST /api/lens/calibrate`** — Calibrates the camera using detected AprilTags and their known physical positions.
  ```json
  { "tags": [{"id": 0, "physical_x": 0, "physical_y": 0, "size_mm": 20, "anchor": "center"}] }
  ```
- **`GET /api/lens/calibration`** — Returns the current calibration status, stored homography matrix, and tag metadata.
- **`GET /api/lens/calibration/check`** — Live quality check. Detects visible AprilTags in the current frame and returns per-tag reprojection error in mm. Saves a debug image viewable at `/api/lens/calibration/debug-image`.

### 3. Detection
- **`GET /api/lens/detect`** — Returns detected workpieces with physical boundaries, segmentation polygon, orientation angle, and area. Saves a debug image viewable at `/api/lens/detect/debug-image`.
  ```json
  {
    "id": "wp_000",
    "class": "workpiece",
    "confidence": 1.0,
    "angle_deg": 12.3,
    "corners_mm": [[x,y],[x,y],[x,y],[x,y]],
    "segmentation_mm": [[x,y], "..."],
    "box_mm": [x1, y1, x2, y2],
    "area_mm2": 9800.5
  }
  ```
- **`GET /api/lens/stream`** — Real-time MJPEG preview stream (~25 FPS).
- **`GET /api/lens/frame`** — Captures and returns a single JPEG frame.

### 4. Debug Images
- **`GET /api/lens/calibration/debug-image`** — Returns the annotated debug JPEG from the last calibration check.
- **`GET /api/lens/detect/debug-image`** — Returns the annotated debug JPEG from the last detection run.

### 5. Transformation (Tap-to-Place)
- **`POST /api/lens/transform`** — Calculates alignment parameters for placing a design on a detected workpiece.
  - **Inputs**: `workpiece_id`, plus either `design_width_mm`/`design_height_mm` OR `design_file` + `dpi`.
  - **Returns**: Rotation (deg), Scale, and Translation (mm).

---

## Production (Raspberry Pi 5 — Native)

### Prerequisites

Ensure the following are installed on the Pi before running `setup_native.sh`:
- HailoRT PCIe Driver (`.deb`) — from [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- HailoRT (`.deb`) — from Hailo Developer Zone
- HailoRT Python Bindings (`.whl`) — from Hailo Developer Zone

> If you are running **without** the Hailo NPU (software-only mode), none of the above are required.

### Setup

```bash
git clone <this-repo>
cd NeonBeam-Lens
cp .env.example .env
# Edit .env for your hardware (camera, bed size, HDR, etc.)
./setup_native.sh
```

`setup_native.sh` will:
1. Install system dependencies (including `libcamera`, `picamera2`)
2. Create a Python virtual environment (`--system-site-packages` for Hailo/Picamera2 access)
3. Install Python requirements
4. Create calibration data directories
5. Install `hailo-apps` and symlink `yolov8s_seg.hef` into `app/models/`
6. Register and enable the `neonbeam-lens` systemd service

### Start / Monitor the Service

```bash
sudo systemctl start neonbeam-lens
sudo systemctl status neonbeam-lens
journalctl -u neonbeam-lens -f
```

---

## Development (CPU Fallback, No Camera Hardware)

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### Running Locally

```bash
cp .env.example .env
# Set USE_HAILO=False and VISION_CAMERA_ID=0 (USB webcam)
uvicorn app.main:app --port 8001 --reload
```

When `USE_HAILO=False`, the service uses the OpenCV morphological contour detector. This works well for high-contrast workpieces (light wood on a dark honeycomb bed) and is suitable for API integration testing.

---

## Calibration Workflow

1. **Print AprilTags**: Use `/api/apriltag/batch` to download a PDF of tags. Print tags `0`–`3` (or more) and attach them to the four corners of your laser bed.
2. **Measure positions**: Note the physical X, Y position of each tag's center (or corner, depending on your `anchor` setting) in mm relative to your bed's (0,0) origin.
3. **Calibrate**: POST to `/api/lens/calibrate` with the tag IDs and their measured physical positions.
4. **Verify**: GET `/api/lens/calibration/check` and inspect the `per_tag_errors`. A `mean_error_mm` under 5mm is considered good. Download the debug image to visually inspect tag detection.

---

## Project Layout

```
app/
├── main.py             # FastAPI entry point & routes
├── services/
│   ├── camera.py       # Frame acquisition, HDR, streaming (Picamera2 / OpenCV)
│   ├── calibration.py  # AprilTag detection & homography calibration
│   ├── inference.py    # NPU inference & OpenCV fallback workpiece detection
│   └── transform.py    # Auto-layout calculation engine
├── models/             # Hailo .hef model files (see models/README.md)
calibration_data/       # Stored homography matrix (matrix.json) & debug images
```
