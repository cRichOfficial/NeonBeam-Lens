import os
import logging
import numpy as np
from typing import List, Dict, Optional
from .calibration import calibration_service

logger = logging.getLogger("vision_api.inference")

class InferenceService:
    def __init__(self):
        self.use_hailo = os.getenv("USE_HAILO", "False").lower() == "true"
        self.hailo_device = None
        self.network_group = None
        self.initialized = False
        
        if self.use_hailo:
            self._init_hailo()
        else:
            logger.info("NPU inference disabled, using software fallback.")
            self.initialized = True

    def _init_hailo(self):
        """Initialize Hailo-8L NPU."""
        try:
            from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, 
                                       InferVStreams, ConfigureParams, InputVStreamParams, 
                                       OutputVStreamParams, FormatType)
            
            # Check for devices
            target_devices = Device.scan()
            if not target_devices:
                raise RuntimeError("No Hailo devices found while USE_HAILO=True")
            
            # In a real implementation, we would load the HEF here
            # model_path = "app/models/workpiece_detector.hef"
            # if not os.path.exists(model_path):
            #     raise FileNotFoundError(f"Hailo model not found at {model_path}")
            
            logger.info(f"Hailo NPU detected: {target_devices}")
            # Stub for real initialization logic
            self.initialized = True
            
        except ImportError:
            if self.use_hailo:
                raise ImportError("hailo_platform library not found but USE_HAILO=True")
        except Exception as e:
            if self.use_hailo:
                raise RuntimeError(f"Failed to initialize Hailo NPU: {e}")

    def detect_workpieces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect workpieces in the image.
        Returns a list of dicts with 'id', 'class', 'confidence', 'box_px', 'corners_px', 'corners_mm'.
        """
        if not self.initialized:
            return []

        # If using Hailo, we would run inference here.
        # For now, we'll use a placeholder or mock.
        if self.use_hailo:
            # results = self._run_hailo_inference(image)
            results = self._mock_detection(image)
        else:
            results = self._mock_detection(image)

        # Map pixel coordinates to physical coordinates
        for res in results:
            corners_px = np.array(res["corners_px"])
            corners_mm = calibration_service.map_pixels_to_mm(corners_px)
            res["corners_mm"] = corners_mm.tolist()
            
            # Also calculate a simple bounding box in mm
            res["box_mm"] = [
                float(corners_mm[:, 0].min()),
                float(corners_mm[:, 1].min()),
                float(corners_mm[:, 0].max()),
                float(corners_mm[:, 1].max())
            ]

        return results

    def _mock_detection(self, image: np.ndarray) -> List[Dict]:
        """A simple mock detector for development."""
        # In a real fallback, we might use OpenCV contour detection to find the workpiece
        # For now, return a single mock workpiece if the image is valid
        if image is None:
            return []
            
        h, w = image.shape[:2]
        # Return a mock workpiece in the center
        return [{
            "id": "wp_001",
            "class": "workpiece",
            "confidence": 0.95,
            "box_px": [w//4, h//4, 3*w//4, 3*h//4],
            "corners_px": [
                [w//4, h//4], [3*w//4, h//4], 
                [3*w//4, 3*h//4], [w//4, 3*h//4]
            ]
        }]

# Global instance
inference_service = InferenceService()
