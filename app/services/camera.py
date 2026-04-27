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
        
        if HAS_PICAMERA2:
            try:
                self.picam2 = Picamera2()
                # Configure for 1280x720 BGR (OpenCV format)
                config = self.picam2.create_video_configuration(main={"format": "BGR888", "size": (1280, 720)})
                self.picam2.configure(config)
                self.picam2.start()
                logger.info("Picamera2 started successfully")
            except Exception as e:
                logger.error(f"Failed to start Picamera2: {e}")
                return False
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id} via OpenCV")
                return False
            logger.info("OpenCV VideoCapture started (fallback)")
        
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
                        self.last_frame = frame
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
