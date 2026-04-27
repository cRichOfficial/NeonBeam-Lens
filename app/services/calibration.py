import cv2
import numpy as np
import json
import os
import logging
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("vision_api.calibration")

class CalibrationService:
    def __init__(self, data_path: str = "calibration_data/matrix.json"):
        self.data_path = data_path
        self.homography_matrix = None
        self.load_calibration()
        
        # AprilTag setup (using Aruco module)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def load_calibration(self):
        """Load calibration data from JSON file."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.homography_matrix = np.array(data['matrix'])
                logger.info(f"Loaded calibration from {self.data_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")

    def save_calibration(self, matrix: np.ndarray):
        """Save calibration matrix to JSON file."""
        self.homography_matrix = matrix
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        try:
            with open(self.data_path, 'w') as f:
                json.dump({'matrix': matrix.tolist()}, f)
            logger.info(f"Saved calibration to {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def detect_tags(self, image: np.ndarray) -> List[Dict]:
        """Detect AprilTags in the image."""
        corners, ids, rejected = self.detector.detectMarkers(image)
        results = []
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                tag_corners = corners[i][0] # 4x2 array of corners
                results.append({
                    "id": int(tag_id),
                    "corners": tag_corners.tolist()
                })
        return results

    def calibrate(self, detected_tags: List[Dict], physical_data: List[Dict]):
        """
        Compute homography matrix from detected tags and their physical locations.
        physical_data: list of {'id': 1, 'x': 0, 'y': 0, 'anchor': 'center', 'size_mm': 50}
        """
        src_pts = []
        dst_pts = []
        
        # Create map for quick lookup
        phys_map = {item['id']: item for item in physical_data}
        
        for tag in detected_tags:
            tid = tag['id']
            if tid in phys_map:
                p = phys_map[tid]
                corners = np.array(tag['corners'])
                
                # If anchor is center, we map the center of the corners to (p['x'], p['y'])
                # However, for better accuracy, we should map all 4 corners if size_mm is known.
                if 'size_mm' in p:
                    s = p['size_mm'] / 2.0
                    # Physical corners relative to center
                    # Aruco corners order: TL, TR, BR, BL
                    target_corners = [
                        [p['x'] - s, p['y'] - s], # TL
                        [p['x'] + s, p['y'] - s], # TR
                        [p['x'] + s, p['y'] + s], # BR
                        [p['x'] - s, p['y'] + s]  # BL
                    ]
                    src_pts.extend(corners)
                    dst_pts.extend(target_corners)
                else:
                    # Fallback to center point if size unknown
                    center = corners.mean(axis=0)
                    src_pts.append(center)
                    dst_pts.append([p['x'], p['y']])
        
        if len(src_pts) < 4:
            raise ValueError("Need at least 4 points (1 tag with size or 4 center points) for calibration")
            
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        matrix, status = cv2.findHomography(src_pts, dst_pts)
        if matrix is not None:
            self.save_calibration(matrix)
            return matrix
        return None

    def map_pixels_to_mm(self, points_px: np.ndarray) -> np.ndarray:
        """Transform pixel coordinates to physical mm."""
        if self.homography_matrix is None:
            return points_px # No calibration
            
        # points_px should be shape (N, 2)
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def get_undistorted_view(self, image: np.ndarray, size_mm: Tuple[int, int] = (500, 500)) -> np.ndarray:
        """Returns a de-warped top-down view of the workspace."""
        if self.homography_matrix is None:
            return image
            
        # For a fixed 1mm per pixel output (e.g. 500x500mm -> 500x500px)
        # We need the inverse matrix to warp the image back
        # But homography_matrix maps px -> mm.
        # So we need to warp the image using the inverse (mm -> px).
        h_inv = np.linalg.inv(self.homography_matrix)
        warped = cv2.warpPerspective(image, h_inv, size_mm)
        return warped

class AprilTagGenerator:
    @staticmethod
    def generate(tag_id: int, size_mm: float = 50.0, dpi: int = 300) -> bytes:
        """
        Generate an AprilTag image with embedded DPI metadata for scaling.
        Returns bytes of a PNG image.
        """
        # Calculate pixel size based on mm and DPI
        # 1 inch = 25.4 mm
        size_px = int((size_mm / 25.4) * dpi)
        
        # AprilTags have a border, so we generate the marker slightly smaller 
        # or just use the full size_px for the marker itself.
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size_px)
        
        # Convert to PIL Image to set DPI metadata
        pil_img = Image.fromarray(tag_img)
        
        # Add a white border (standard for AprilTags to ensure detection)
        border_px = size_px // 10
        total_size = size_px + 2 * border_px
        final_img = Image.new("L", (total_size, total_size), 255)
        final_img.paste(pil_img, (border_px, border_px))
        
        # Save to buffer with DPI
        buf = io.BytesIO()
        final_img.save(buf, format="PNG", dpi=(dpi, dpi))
        return buf.getvalue()

# Global instance
calibration_service = CalibrationService()
