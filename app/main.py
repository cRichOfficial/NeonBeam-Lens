import asyncio
import logging
import sys
import io
import cv2
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .services.mdns_advertiser import MdnsAdvertiser
from .services.camera import camera_service
from .services.calibration import (
    calibration_service, AprilTagGenerator, CheckerboardGenerator,
    LensCalibrationSession,
)
from .services.inference import inference_service
from .services.transform import transform_service

# Robust Debug Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vision_api")

# Silence noisy external libraries
logging.getLogger("picamera2").setLevel(logging.WARNING)
logging.getLogger("libcamera").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("NeonBeam Lens starting up...")
    
    # Start Camera
    camera_service.start()

    # Advertise on the LAN via mDNS
    mdns = MdnsAdvertiser()
    await asyncio.to_thread(mdns.start)

    yield

    logger.debug("NeonBeam Lens shutting down...")
    camera_service.stop()
    await asyncio.to_thread(mdns.stop)


app = FastAPI(title="NeonBeam Lens API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class CalibrationPoint(BaseModel):
    id: int
    physical_x: float
    physical_y: float
    size_mm: Optional[float] = None
    anchor: str = "center"

class CalibrationRequest(BaseModel):
    tags: List[CalibrationPoint]

class TransformRequest(BaseModel):
    workpiece_id: str
    design_width_mm: Optional[float] = None
    design_height_mm: Optional[float] = None
    dpi: Optional[float] = None
    padding_mm: float = 5.0
    # For file uploads, we'll handle separately

# Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "machine_vision",
        "detector": "opencv",
        "camera_active": camera_service.running,
        "lens_calibrated": calibration_service.lens_calibrated,
        "homography_calibrated": calibration_service.homography_matrix is not None,
        "reference_frame_captured": inference_service.has_reference,
    }

@app.get("/api/apriltag/generate/{tag_id}")
async def generate_tag(tag_id: int, size_mm: float = 50.0, dpi: int = 300, guide_distance_mm: float = 0.0):
    """Generate a single tag with physical sizing metadata."""
    tag_bytes = AprilTagGenerator.generate(tag_id, size_mm, dpi, guide_distance_mm=guide_distance_mm)
    return Response(content=tag_bytes, media_type="application/pdf")

@app.get("/api/apriltag/batch")
async def batch_generate_tags(start_id: int = 0, count: int = 4, size_mm: float = 50.0, dpi: int = 300, paper_width_in: float = 8.5, paper_height_in: float = 11.0, guide_distance_mm: float = 0.0):
    """Return a multi-page PDF containing all requested tags packed for standard printing."""
    pdf_bytes = AprilTagGenerator.generate_batch_document(start_id, count, size_mm, dpi, paper_width_in, paper_height_in, guide_distance_mm=guide_distance_mm)
    return Response(content=pdf_bytes, media_type="application/pdf")

# ── Checkerboard Generation ──────────────────────────────────────────────────

@app.get("/api/lens/checkerboard/generate")
async def generate_checkerboard(
    rows: int = 9, cols: int = 6, square_mm: float = 25.0, dpi: int = 300,
):
    """Generate a printable checkerboard PDF for lens distortion calibration."""
    pdf_bytes = CheckerboardGenerator.generate(rows, cols, square_mm, dpi)
    return Response(content=pdf_bytes, media_type="application/pdf")

# ── Lens Distortion Calibration (guided multi-step) ──────────────────────────

# Module-level session holder (one active session at a time)
_lens_session: LensCalibrationSession | None = None
_lens_capture_lock = asyncio.Lock()

@app.post("/api/lens/calibrate-lens/start")
async def lens_calibration_start(
    rows: int = 9, cols: int = 6, square_mm: float = 25.0,
):
    """Start a new guided lens calibration session."""
    global _lens_session
    _lens_session = LensCalibrationSession(rows=rows, cols=cols, square_mm=square_mm)
    return _lens_session.start()

@app.post("/api/lens/calibrate-lens/capture")
async def lens_calibration_capture():
    """Capture a frame and attempt to detect the checkerboard.

    The CV computation (findChessboardCorners) is offloaded to a thread pool
    so the FastAPI event loop is not blocked during the ~0.5-2s detection.
    A lock prevents overlapping captures if the client calls rapidly.
    """
    if _lens_session is None or not _lens_session.active:
        raise HTTPException(status_code=400, detail="No active lens calibration session. Call /start first.")
    if _lens_capture_lock.locked():
        raise HTTPException(status_code=429, detail="A capture is already in progress. Please wait.")
    async with _lens_capture_lock:
        frame = camera_service.get_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="Camera frame not available.")
        return await asyncio.to_thread(_lens_session.capture, frame)

@app.post("/api/lens/calibrate-lens/finish")
async def lens_calibration_finish():
    """Compute lens calibration from captured frames and save K/D coefficients."""
    global _lens_session
    if _lens_session is None or not _lens_session.active:
        raise HTTPException(status_code=400, detail="No active lens calibration session.")
    result = await asyncio.to_thread(_lens_session.finish, calibration_service)
    _lens_session = None

    # The undistortion map has changed, so the stored reference frame is now
    # spatially misaligned. Delete it — it will be recaptured automatically
    # the next time the user runs the mapping calibration.
    inference_service.clear_reference_frame()
    logger.info("Lens recalibration complete — reference frame invalidated.")

    return result

@app.delete("/api/lens/calibrate-lens")
async def lens_calibration_reset():
    """Delete the saved lens calibration, forcing a full recalibration.

    Also cancels any active session. After this call, lens_calibrated will
    be False on the next health check.
    """
    global _lens_session
    _lens_session = None  # kill any active session

    lens_path = calibration_service.lens_data_path
    if os.path.exists(lens_path):
        os.remove(lens_path)

    # Clear in-memory state
    calibration_service.camera_matrix = None
    calibration_service.dist_coeffs = None
    calibration_service.lens_calibrated = False
    calibration_service._undistort_map1 = None
    calibration_service._undistort_map2 = None
    calibration_service._undistort_resolution = None

    return {"status": "reset", "message": "Lens calibration cleared. Run /start to recalibrate."}

@app.get("/api/lens/calibrate-lens/status")
async def lens_calibration_status():
    """Get the current lens calibration session status."""
    if _lens_session is None or not _lens_session.active:
        # Check if a saved calibration exists
        if calibration_service.lens_calibrated:
            return {
                "status": "calibrated",
                "message": "Lens calibration is loaded from disk. No active session.",
                "lens_model": "fisheye" if calibration_service.is_fisheye else "standard",
            }
        return {"status": "uncalibrated", "message": "No active session and no saved lens calibration."}
    return _lens_session._status()

@app.get("/api/lens/calibrate-lens/preview")
async def lens_calibration_preview():
    """Returns the preview image from the last lens calibration capture."""
    debug_path = "calibration_data/lens_calibration_preview.jpg"
    if not os.path.exists(debug_path):
        raise HTTPException(status_code=404, detail="No preview available. Run a capture first.")
    return FileResponse(debug_path, media_type="image/jpeg")

@app.get("/api/lens/calibration/debug-image")
async def get_calibration_debug_image():
    """Returns the visual debug image from the last calibration check."""
    debug_path = "calibration_data/calibration_debug.jpg"
    if not os.path.exists(debug_path):
        raise HTTPException(status_code=404, detail="Debug image not found. Run /api/lens/calibration/check first.")
    return FileResponse(debug_path, media_type="image/jpeg")

@app.get("/api/lens/detect/debug-image")
async def get_detect_debug_image():
    """Returns the visual debug image from the last object detection run."""
    debug_path = "calibration_data/detect_debug.jpg"
    if not os.path.exists(debug_path):
        raise HTTPException(status_code=404, detail="Detection debug image not found. Run /api/lens/detect first.")
    return FileResponse(debug_path, media_type="image/jpeg")

@app.post("/api/lens/calibrate")
async def calibrate(request: CalibrationRequest):
    frame = camera_service.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame not available")

    detected = calibration_service.detect_tags(frame)
    if not detected:
        return {"status": "error", "message": "No AprilTags detected in frame"}

    physical_data = [t.dict() for t in request.tags]
    for p in physical_data:
        p['x'] = p.pop('physical_x')
        p['y'] = p.pop('physical_y')

    try:
        matrix = calibration_service.calibrate(detected, physical_data)

        # Capture the empty-bed reference frame for background subtraction.
        # At this point the bed must be clear (only AprilTags present), making
        # it the ideal moment to save the reference without a separate workflow.
        ref_frame = camera_service.get_frame()
        if ref_frame is not None:
            undistorted = calibration_service.undistort(ref_frame)
            await asyncio.to_thread(inference_service.save_reference_frame, undistorted)
            logger.info("Empty-bed reference frame captured during mapping calibration.")
        else:
            logger.warning("Could not capture reference frame — camera returned None.")

        return {
            "status": "calibrated",
            "detected_count": len(detected),
            "reference_frame_captured": inference_service.has_reference,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/lens/reference-frame/status")
async def reference_frame_status():
    """Returns whether a background reference frame has been captured."""
    import os, time
    path = inference_service._REFERENCE_PATH
    exists = os.path.exists(path)
    captured_at = None
    if exists:
        captured_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(path))
        )
    return {"captured": exists, "captured_at": captured_at}


@app.delete("/api/lens/reference-frame")
async def delete_reference_frame():
    """Manually invalidate the background reference frame."""
    had = inference_service.has_reference
    inference_service.clear_reference_frame()
    return {"status": "deleted" if had else "not_found"}


@app.get("/api/lens/calibration")
async def get_calibration():
    """Retrieve current calibration status and stored parameters."""
    return calibration_service.get_status()

@app.get("/api/lens/calibration/check")
async def check_calibration():
    """
    Live calibration quality check.
    Detects visible AprilTags in the current frame, maps them through the
    homography, and returns per-tag reprojection error in mm.
    """
    frame = camera_service.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame not available")
    return calibration_service.check_calibration(frame)

@app.get("/api/lens/detect")
async def detect_objects():
    frame = camera_service.get_frame()
    if frame is None:
        return {"status": "error", "message": "Camera offline"}
    
    workpieces = inference_service.detect_workpieces(frame)
    return {
        "status": "ok", 
        "workpieces": workpieces,
        "debug_image_url": "/api/lens/detect/debug-image"
    }

@app.post("/api/lens/transform")
async def calculate_transform(
    workpiece_id: str = Form(...),
    design_width_mm: Optional[float] = Form(None),
    design_height_mm: Optional[float] = Form(None),
    dpi: Optional[float] = Form(None),
    padding_mm: float = Form(5.0),
    design_file: Optional[UploadFile] = File(None)
):
    # 1. Get current detection for the workpiece
    frame = camera_service.get_frame()
    workpieces = inference_service.detect_workpieces(frame)
    
    wp = next((w for w in workpieces if w["id"] == workpiece_id), None)
    if not wp:
        # Fallback: if no specific ID match, use the first one if only one exists
        if len(workpieces) == 1:
            wp = workpieces[0]
        else:
            raise HTTPException(status_code=404, detail="Workpiece not found")

    # 2. Prepare design data
    design_data = {}
    if design_width_mm and design_height_mm:
        design_data = {"width_mm": design_width_mm, "height_mm": design_height_mm}
    elif design_file and dpi:
        file_bytes = await design_file.read()
        design_data = {"image_data": file_bytes, "dpi": dpi}
    else:
        raise HTTPException(status_code=400, detail="Missing design dimensions or image + DPI")

    # 3. Calculate transform
    result = transform_service.calculate_transform(design_data, wp, padding_mm)
    return result

@app.get("/api/lens/stream")
async def stream_video():
    """MJPEG streaming endpoint."""
    async def frame_generator():
        while True:
            frame = camera_service.get_frame()
            if frame is not None:
                # Optional: apply undistortion for the preview
                # frame = calibration_service.get_undistorted_view(frame)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.04) # ~25 FPS

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/lens/frame")
async def get_frame():
    """Return a single frame as JPEG."""
    jpeg = camera_service.get_jpeg_frame()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Camera frame not available")
    return Response(content=jpeg, media_type="image/jpeg")
