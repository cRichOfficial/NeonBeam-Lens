# NeonBeam Lens
### Machine Vision Layer — Workpiece Detection & Camera Calibration

NeonBeam Lens is a Python FastAPI service designed for the **Raspberry Pi 5**. It provides the "eyes" for the NeonBeam laser engraver suite, handling real-time workpiece detection, camera-to-bed calibration, and automatic design alignment.

---

## Features

- **AprilTag Calibration**: Automatic mapping of camera pixels to physical laser bed coordinates (mm) using standard AprilTags placed at known positions on the bed corners.
- **Texture-Based Workpiece Detection**: Detects objects of **any colour** (bright wood, dark slate, clear acrylic) on the black honeycomb bed using local-variance texture analysis. The honeycomb's repeating hole pattern has high local pixel variance; any solid workpiece has low variance.
- **Multi-Object Support**: Detects and returns all workpieces simultaneously, each with physical boundaries, orientation angle, segmentation polygon, and area in mm².
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

## How Detection Works

The detector uses **local texture variance** to distinguish workpieces from the honeycomb bed:

1. **Downsample** the camera frame to `DETECT_MAX_DIM` px on the long side (default 1280).
2. **Preprocess**: Convert to grayscale, apply CLAHE contrast enhancement, and Gaussian blur to smooth away per-pixel noise while preserving the honeycomb texture pattern.
3. **Build ROI**: Detect AprilTags on the full-resolution image (where the ArUco parameters are tuned) and construct a convex hull mask. Only the area inside the tags is searched — the laser frame, table, and cables are excluded.
4. **Compute local variance**: `Var(X) = E[X²] − E[X]²` in a sliding window of `DETECT_VARIANCE_WINDOW` pixels. This is implemented with two fast box filters.
5. **Threshold**: The honeycomb (repeating holes) has high variance; a solid workpiece has low variance. We estimate the honeycomb baseline from the 60th percentile of in-ROI variance and threshold at 40% of that value.
6. **Morphological close + dilate** to merge gaps (wood grain, knots, honeycomb shadows at edges).
7. **Find contours**, scale to full resolution, compute oriented bounding rect + convex hull.
8. **Map to mm** via the calibration homography and apply physical area filtering.

This approach detects workpieces of **any brightness** — bright wood, white paper, dark slate, and clear acrylic all produce low local variance compared to the honeycomb.

---

## Environment Variables

### Camera

| Variable | Default | Description |
|---|---|---|
| `VISION_CAMERA_ID` | `0` | OpenCV camera index (used when Picamera2 is unavailable). |
| `CAMERA_RESOLUTION` | `max` | Capture resolution. `max` uses the sensor's full native resolution. Override with `WIDTHxHEIGHT` (e.g. `1920x1080`). |
| `CAMERA_HDR` | `False` | Enable sensor-level hardware HDR (IMX708/Camera v3 only). Caps resolution to ~3MP (2304×1296). |
| `FISHEYE_LENS` | `False` | Lens distortion model. `False` = standard pinhole (≤120° HFOV). `True` = Kannala-Brandt fisheye (>140° HFOV). Used during guided lens calibration. |
| `WORKSPACE_HEIGHT_MM` | `500` | Physical height of the laser bed in mm. Used to flip the Y-axis so (0,0) is bottom-left. Set to `0` to disable Y-flip. |

### Auto-Exposure Settling

| Variable | Default | Description |
|---|---|---|
| `MIN_FRAME_BRIGHTNESS` | `20` | Mean pixel brightness below which a frame is considered too dark. |
| `AE_SETTLE_RETRIES` | `10` | Maximum retry attempts while waiting for AE to stabilise. |
| `AE_SETTLE_DELAY` | `0.1` | Seconds between AE retry attempts. Total max wait = retries × delay. |

### Detection

| Variable | Default | Description |
|---|---|---|
| `DETECT_MAX_DIM` | `1280` | Maximum long-side resolution (px) for the detection pipeline. See below. |
| `DETECT_VARIANCE_WINDOW` | `25` | Local-variance window size (px at 720p reference). Must span several honeycomb holes. |
| `MIN_WORKPIECE_AREA_MM2` | `500` | Minimum physical area (mm²) to report as a workpiece. |
| `MAX_WORKPIECE_AREA_MM2` | `160000` | Maximum physical area (mm²). Filters out the bed frame itself. |
| `HONEYCOMB_KERNEL_SIZE` | `25` | Morphological kernel (px at 720p ref) for bridging gaps within a workpiece. |
| `APRILTAG_MASK_PADDING` | `30` | Extra pixels (720p ref) to expand the AprilTag exclusion zone. |

### Logging

| Variable | Default | Description |
|---|---|---|
| `VISION_DEBUG_LOGGING` | `True` | `True` = DEBUG level (frame timings, variance stats). `False` = INFO only. |

---

### About `DETECT_MAX_DIM`

The detection pipeline downsamples the camera frame to a capped working resolution before any expensive processing. Only the detection pipeline uses this downsampled copy; the original full-resolution frame is preserved. Detected workpiece contours are scaled back to full-resolution coordinates before the calibration homography maps them to physical mm values, so **accuracy is not affected**.

| `DETECT_MAX_DIM` | Working size (approx.) | Detection time (Pi 5) | Use when… |
|---|---|---|---|
| `640` | 640×360 | ~0.1s | Fastest; good for large workpieces only |
| `1280` *(default)* | 1280×720 | ~0.3s | Best balance for most workpieces |
| `1920` | 1920×1080 | ~0.8s | Finer contour detail for small/intricate items |

---

## API Endpoints

### 1. AprilTag Generation
- **`GET /api/apriltag/generate/{tag_id}`** — Returns a printable AprilTag as a PDF (includes cut-line border and ID label).
- **`GET /api/apriltag/batch`** — Returns a multi-page PDF of tags densely packed for printing.
  - Params: `start_id`, `count`, `size_mm`, `dpi`, `paper_width_in`, `paper_height_in`

### 2. Lens Distortion Calibration (Guided)
- **`GET /api/lens/checkerboard/generate`** — Generate a printable checkerboard PDF for lens calibration.
  - Params: `rows` (default 9), `cols` (default 6), `square_mm` (default 25), `dpi` (default 300)
- **`POST /api/lens/calibrate-lens/start`** — Start a guided lens calibration session.
  - Params: `rows`, `cols`, `square_mm` (must match your printed checkerboard)
- **`POST /api/lens/calibrate-lens/capture`** — Capture a frame and detect the checkerboard. Returns placement guidance for the next capture.
- **`POST /api/lens/calibrate-lens/finish`** — Compute and save the lens calibration (K/D coefficients).
- **`GET /api/lens/calibrate-lens/status`** — Get session progress or saved calibration status.
- **`GET /api/lens/calibrate-lens/preview`** — Preview image from the last capture attempt with detected corners drawn.

### 3. Homography Calibration (AprilTags)
- **`POST /api/lens/calibrate`** — Calibrates the camera using detected AprilTags and their known physical positions.
  ```json
  { "tags": [{"id": 0, "physical_x": 0, "physical_y": 0, "size_mm": 20, "anchor": "center"}] }
  ```
- **`GET /api/lens/calibration`** — Returns the current calibration status, stored homography matrix, and tag metadata.
- **`GET /api/lens/calibration/check`** — Live quality check. Detects visible AprilTags in the current frame and returns per-tag reprojection error in mm.

### 4. Detection
- **`GET /api/lens/detect`** — Returns all detected workpieces with physical boundaries, segmentation polygon, orientation angle, and area.
  ```json
  {
    "workpieces": [
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
    ]
  }
  ```
- **`GET /api/lens/stream`** — Real-time MJPEG preview stream (~25 FPS).
- **`GET /api/lens/frame`** — Captures and returns a single JPEG frame.

### 5. Debug Images
- **`GET /api/lens/calibration/debug-image`** — Annotated debug JPEG from the last calibration check.
- **`GET /api/lens/detect/debug-image`** — Annotated debug JPEG from the last detection run.

### 6. Transformation (Tap-to-Place)
- **`POST /api/lens/transform`** — Calculates alignment parameters for placing a design on a detected workpiece.
  - **Inputs**: `workpiece_id`, plus either `design_width_mm`/`design_height_mm` OR `design_file` + `dpi`.
  - **Returns**: Rotation (deg), Scale, and Translation (mm).

---

## Production (Raspberry Pi 5 — Native)

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
2. Create a Python virtual environment (`--system-site-packages` for Picamera2 access)
3. Install Python requirements
4. Create calibration data directories
5. Register and enable the `neonbeam-lens` systemd service

### Start / Monitor the Service

```bash
sudo systemctl start neonbeam-lens
sudo systemctl status neonbeam-lens
journalctl -u neonbeam-lens -f
```

---

## Development (No Camera Hardware)

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### Running Locally

```bash
cp .env.example .env
# Set VISION_CAMERA_ID=0 (USB webcam or laptop camera)
uvicorn app.main:app --port 8001 --reload
```

---

## Calibration Workflow

Calibration is a two-stage process. Lens calibration is a **one-time** setup per camera. AprilTag calibration should be re-run whenever the camera or tags are moved.

### Stage 1: Lens Distortion Calibration (one-time)

1. **Print checkerboard**: `GET /api/lens/checkerboard/generate` → print the PDF at 100% scale.
2. **Tape to flat surface**: Attach the printout to a rigid flat surface (clipboard, book, cardboard).
3. **Start session**: `POST /api/lens/calibrate-lens/start`
4. **Capture frames**: Hold the checkerboard in front of the camera and call `POST /api/lens/calibrate-lens/capture`. The API returns guidance on where to place the board next (9 zones across the frame).
5. **Repeat** for 10–15 captures, covering at least 5 zones and varying the board angle slightly.
6. **Finish**: `POST /api/lens/calibrate-lens/finish` → computes K/D coefficients and saves them to `calibration_data/lens_calibration.json`.
7. The calibration persists across restarts. You only need to redo this if you change the camera or lens.

### Stage 2: AprilTag Homography Calibration

1. **Print AprilTags**: Use `/api/apriltag/batch` to download a PDF of tags. Print tags `0`–`3` (or more) and attach them to the four corners of your laser bed.
2. **Measure positions**: Note the physical X, Y position of each tag's center (or corner, depending on your `anchor` setting) in mm relative to your bed's (0,0) origin.
3. **Calibrate**: POST to `/api/lens/calibrate` with the tag IDs and their measured physical positions.
4. **Verify**: GET `/api/lens/calibration/check` and inspect the `per_tag_errors`. A `mean_error_mm` under 5mm is considered good.

> **Important:** Always run the lens calibration (Stage 1) **before** the AprilTag calibration (Stage 2). Undistortion changes pixel positions, so the homography must be computed on undistorted images.

---

## Project Layout

```
app/
├── main.py             # FastAPI entry point & routes
├── services/
│   ├── camera.py       # Frame acquisition, HDR, AE settling (Picamera2 / OpenCV)
│   ├── calibration.py  # AprilTag detection & homography calibration
│   ├── inference.py    # Texture-variance workpiece detection
│   └── transform.py    # Auto-layout calculation engine
calibration_data/       # Stored homography matrix (matrix.json) & debug images
```
