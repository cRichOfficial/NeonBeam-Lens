import cv2
import threading
import time
import logging
import os

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
        self.cap = None
        self.last_frame = None
        self.running = False
        self.thread = None
        self._initialized = True
        logger.info(f"CameraService initialized with camera ID {self.camera_id}")

    def start(self):
        """Start the background frame capture thread."""
        if self.running:
            return
        
        # Explicitly use V4L2 backend for Linux/Pi 5 compatibility
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            logger.warning(f"Failed to open camera {self.camera_id} via V4L2, trying default...")
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Camera capture thread started")
        return True

    def stop(self):
        """Stop the background frame capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera capture thread stopped")

    def _capture_loop(self):
        # Optional: Set properties for better compatibility on Pi 5
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # BGR3 is also supported on /dev/video0 and might be more stable
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BGR3'))

        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {width}x{height} @ {fps} FPS")

        fail_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame
                fail_count = 0
            else:
                fail_count += 1
                if fail_count % 30 == 0: # Log every ~3 seconds
                    logger.warning(f"Failed to capture frame (consecutive failures: {fail_count})")
                
                if fail_count > 100:
                    logger.error("Too many capture failures. Restarting camera...")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
                    fail_count = 0
                
                time.sleep(0.1)
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
