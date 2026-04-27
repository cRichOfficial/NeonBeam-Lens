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
        
        # Use default backend when LD_PRELOAD is active for better compatibility
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
        # Let the wrapper negotiate the best format/resolution for now
        # Forcing properties on Pi 5 can cause buffer reshape errors
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera opened: {width}x{height}")

        fail_count = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame = frame
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count % 30 == 0:
                        logger.warning(f"Failed to capture frame (consecutive: {fail_count})")
                    time.sleep(0.1)
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
