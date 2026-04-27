import asyncio
import logging
import sys
import io
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .services.mdns_advertiser import MdnsAdvertiser
from .services.camera import camera_service
from .services.calibration import calibration_service, AprilTagGenerator
from .services.inference import inference_service
from .services.transform import transform_service

# Robust Debug Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vision_api")


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
        "hailo_npu": inference_service.initialized,
        "camera_active": camera_service.running
    }

@app.get("/api/apriltag/generate/{tag_id}")
async def generate_tag(tag_id: int, size_px: int = 400):
    tag_img = AprilTagGenerator.generate(tag_id, size_px)
    _, buffer = cv2.imencode('.png', tag_img)
    return Response(content=buffer.tobytes(), media_type="image/png")

@app.post("/api/lens/calibrate")
async def calibrate(request: CalibrationRequest):
    frame = camera_service.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera frame not available")
    
    detected = calibration_service.detect_tags(frame)
    if not detected:
        return {"status": "error", "message": "No AprilTags detected in frame"}
    
    physical_data = [t.dict() for t in request.tags]
    # Rename fields to match calibration service expectations
    for p in physical_data:
        p['x'] = p.pop('physical_x')
        p['y'] = p.pop('physical_y')

    try:
        matrix = calibration_service.calibrate(detected, physical_data)
        return {"status": "calibrated", "detected_count": len(detected)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/lens/detect")
async def detect_objects():
    frame = camera_service.get_frame()
    if frame is None:
        return {"status": "error", "message": "Camera offline"}
    
    workpieces = inference_service.detect_workpieces(frame)
    return {"status": "ok", "workpieces": workpieces}

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
