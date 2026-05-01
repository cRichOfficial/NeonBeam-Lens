import cv2
import threading
import time
import logging
import os
import numpy as np

# Try to import Picamera2 for Pi 5 support
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

logger = logging.getLogger("vision_api.camera")

class CameraService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.camera_id = int(os.getenv("VISION_CAMERA_ID", "0"))
        self.picam2 = None
        self.cap = None
        self.last_frame = None
        self.running = False
        self.thread = None
        self._initialized = True
        logger.info(f"CameraService initialized (Has Picamera2: {HAS_PICAMERA2})")

    def start(self):
        """Start the background frame capture thread."""
        if self.running:
            return

        # Optional resolution cap from env: "WIDTHxHEIGHT" e.g. "4056x3040"
        # Defaults to "max" which uses the sensor's full native resolution.
        res_env = os.getenv("CAMERA_RESOLUTION", "max").strip().lower()

        if HAS_PICAMERA2:
            try:
                self.picam2 = Picamera2()

                hdr_env = os.getenv("CAMERA_HDR", "False").strip().lower() == "true"
                controls = {}
                if hdr_env:
                    controls["HdrMode"] = 2  # 2 corresponds to libcamera.controls.HdrModeEnum.Sensor
                    logger.info("Picamera2: Requesting Hardware HDR (Sensor mode)")

                if res_env == "max":
                    # Query the sensor's full pixel array size — the true hardware maximum
                    sensor_res = self.picam2.camera_properties.get("PixelArraySize")
                    if sensor_res:
                        capture_w, capture_h = sensor_res
                        if hdr_env and capture_w > 2304:
                            # IMX708 hardware HDR is limited to ~3MP (2304x1296)
                            capture_w, capture_h = 2304, 1296
                            logger.info(f"Picamera2: Capping max resolution to {capture_w}x{capture_h} for HDR mode")
                        else:
                            logger.info(f"Picamera2: using max sensor resolution {capture_w}x{capture_h}")
                    else:
                        # Fallback: pick the largest mode available
                        modes = self.picam2.sensor_modes
                        if modes:
                            largest = max(modes, key=lambda m: m["size"][0] * m["size"][1])
                            capture_w, capture_h = largest["size"]
                            if hdr_env and capture_w > 2304:
                                capture_w, capture_h = 2304, 1296
                            logger.info(f"Picamera2: largest sensor mode {capture_w}x{capture_h}")
                        else:
                            capture_w, capture_h = 2304 if hdr_env else 4056
                            capture_h = 1296 if hdr_env else 3040  # Pi HQ safe default
                            logger.warning(f"Picamera2: could not detect sensor modes, defaulting to {capture_w}x{capture_h}")
                else:
                    try:
                        capture_w, capture_h = [int(v) for v in res_env.split("x")]
                        logger.info(f"Picamera2: using configured resolution {capture_w}x{capture_h}")
                    except ValueError:
                        logger.warning(f"Invalid CAMERA_RESOLUTION '{res_env}', falling back to max")
                        sensor_res = self.picam2.camera_properties.get("PixelArraySize", (4056, 3040))
                        capture_w, capture_h = sensor_res
                        if hdr_env and capture_w > 2304:
                            capture_w, capture_h = 2304, 1296

                config = self.picam2.create_video_configuration(
                    main={"format": "BGR888", "size": (capture_w, capture_h)},
                    controls=controls
                )
                self.picam2.configure(config)
                self.picam2.start()
                logger.info(f"Picamera2 started at {capture_w}x{capture_h}")
            except Exception as e:
                logger.error(f"Failed to start Picamera2: {e}")
                return False
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id} via OpenCV")
                return False

            if res_env == "max":
                # Request an absurdly large resolution — the driver will clamp to its maximum
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
            elif "x" in res_env:
                try:
                    w, h = [int(v) for v in res_env.split("x")]
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                except ValueError:
                    pass

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"OpenCV VideoCapture started at {actual_w}x{actual_h}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop the background frame capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
        
        if self.cap:
            self.cap.release()
        logger.info("Camera capture stopped")

    def _capture_loop(self):
        while self.running:
            try:
                if self.picam2:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # Picamera2 capture_array() returns RGB regardless of
                        # the BGR888 format config. Convert to BGR so all
                        # downstream OpenCV code and cv2.imwrite work correctly.
                        self.last_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        self.last_frame = frame
            except Exception as e:
                logger.error(f"Error during capture: {e}")
                time.sleep(1.0)
            
            time.sleep(0.01)

    def get_frame(self):
        """Return the most recent frame."""
        return self.last_frame

    def get_jpeg_frame(self):
        """Return the most recent frame as a JPEG encoded byte string."""
        frame = self.get_frame()
        if frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return buffer.tobytes()

# Global instance
camera_service = CameraService()
