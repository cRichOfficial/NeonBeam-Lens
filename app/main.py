import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .services.mdns_advertiser import MdnsAdvertiser

# Robust Debug Logging for Pi 5 NPU Analysis
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vision_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("NeonBeam Lens starting up… Checking Hailo-8L NPU status…")

    # Advertise on the LAN via mDNS so the Discovery Sidecar can find this
    # service automatically.  Silently no-ops on Docker Desktop where multicast
    # doesn't cross the NAT — the sidecar's subnet scan covers that case.
    mdns = MdnsAdvertiser()
    await asyncio.to_thread(mdns.start)

    yield

    logger.debug("NeonBeam Lens shutting down…")
    await asyncio.to_thread(mdns.stop)


app = FastAPI(title="NeonBeam Lens API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CalibrationRequest(BaseModel):
    reference_points: list


class WorkloadTransformRequest(BaseModel):
    workpiece_id: str
    target_pos: dict


@app.get("/api/health")
async def health_check():
    logger.debug("Health check invoked")
    return {"status": "ok", "service": "machine_vision"}

@app.post("/api/calibrate")
async def calibrate_camera(request: CalibrationRequest):
    logger.debug(f"Calibration requested with {len(request.reference_points)} points")
    # Stub: Compute transformation matrix
    return {"status": "calibrated", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

@app.get("/api/feed")
async def get_live_feed():
    logger.debug("Live feed frame requested")
    # Stub: Return frame or WS endpoint for MJPEG stream
    return {"status": "streaming"}

@app.post("/api/transform")
async def calculate_transformation(request: WorkloadTransformRequest):
    logger.debug(f"Transform target calculated for {request.workpiece_id}")
    return {"scale": 1.0, "rotation": 0, "translation": {"x": 10, "y": 20}}
