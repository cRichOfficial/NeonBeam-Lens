import cv2
import numpy as np
import json
import os
import logging
import io
import base64
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple, Union

logger = logging.getLogger("vision_api.calibration")

class CalibrationService:
    def __init__(self, data_path: str = "calibration_data/matrix.json"):
        self.data_path = data_path
        self.homography_matrix = None
        self.calibration_data = None
        
        # Generic Fisheye Calibration for 175 FOV lenses (Disabled by default)
        self.is_fisheye = os.getenv("FISHEYE_LENS", "False").lower() == "true"
        # Standard estimates for a 1/4" sensor with 175 FOV
        self.k = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], dtype=np.float32)
        self.d = np.array([-0.05, -0.01, 0, 0], dtype=np.float32)

        # Y-axis flip: camera sees Y=0 at top, laser bed uses Y=0 at bottom-left.
        # Set WORKSPACE_HEIGHT_MM to the physical height of your bed in mm.
        self.workspace_height_mm = float(os.getenv("WORKSPACE_HEIGHT_MM", "0"))
        
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
                    self.calibration_data = data.get('calibration_data', None)
                logger.info(f"Loaded calibration from {self.data_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")

    def save_calibration(self, matrix: np.ndarray, calibration_data: Dict = None):
        """Save calibration matrix to JSON file."""
        self.homography_matrix = matrix
        self.calibration_data = calibration_data
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        try:
            with open(self.data_path, 'w') as f:
                json.dump({'matrix': matrix.tolist(), 'calibration_data': calibration_data}, f)
            logger.info(f"Saved calibration to {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def detect_tags(self, image: np.ndarray) -> List[Dict]:
        """Detect AprilTags in the image. Automatically undistorts if enabled."""
        # Ensure we are working with an undistorted image for tag detection
        img_processed = self.undistort(image)
        
        corners, ids, rejected = self.detector.detectMarkers(img_processed)
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
                
                # Map corners based on the specified anchor and tag size
                if 'size_mm' in p:
                    s = p['size_mm']
                    half = s / 2.0
                    px, py = p['x'], p['y']
                    anchor = p.get('anchor', 'center').lower().replace('_', '-')
                    
                    # Aruco corners are always returned in order: TL, TR, BR, BL
                    # Physical coordinates use Cartesian convention: Y increases upward.
                    # So 'top' means HIGHER Y value, 'bottom' means LOWER Y value.
                    if anchor == 'center':
                        target_corners = [
                            [px - half, py + half], # TL (left, up)
                            [px + half, py + half], # TR (right, up)
                            [px + half, py - half], # BR (right, down)
                            [px - half, py - half]  # BL (left, down)
                        ]
                    elif anchor in ['top-left', 'tl']:
                        # anchor point is the top-left corner of the tag
                        target_corners = [
                            [px,     py    ],  # TL
                            [px + s, py    ],  # TR
                            [px + s, py - s],  # BR
                            [px,     py - s]   # BL
                        ]
                    elif anchor in ['top-right', 'tr']:
                        # anchor point is the top-right corner of the tag
                        target_corners = [
                            [px - s, py    ],  # TL
                            [px,     py    ],  # TR
                            [px,     py - s],  # BR
                            [px - s, py - s]   # BL
                        ]
                    elif anchor in ['bottom-right', 'br']:
                        # anchor point is the bottom-right corner of the tag
                        target_corners = [
                            [px - s, py + s],  # TL
                            [px,     py + s],  # TR
                            [px,     py    ],  # BR
                            [px - s, py    ]   # BL
                        ]
                    elif anchor in ['bottom-left', 'bl']:
                        # anchor point is the bottom-left corner of the tag
                        target_corners = [
                            [px,     py + s],  # TL
                            [px + s, py + s],  # TR
                            [px + s, py    ],  # BR
                            [px,     py    ]   # BL
                        ]
                    else:
                        # Default to center
                        target_corners = [[px-half, py+half], [px+half, py+half], [px+half, py-half], [px-half, py-half]]
                    
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
        
        # Use RANSAC to filter out noisy tag detections
        matrix, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if matrix is not None:
            calib_data = {
                "detected_tags": detected_tags,
                "physical_data": physical_data
            }
            self.save_calibration(matrix, calib_data)
            return matrix
        return None

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Apply fisheye undistortion if enabled."""
        if not self.is_fisheye or hasattr(image, '_neon_undistorted'):
            return image
        
        h, w = image.shape[:2]
        # Estimate new camera matrix to keep the full FOV
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.k, self.d, (w, h), np.eye(3), balance=1.0)
        undistorted = cv2.fisheye.undistortImage(image, self.k, self.d, Knew=new_k)
        
        # Mark as processed to prevent double-undistortion
        setattr(undistorted, '_neon_undistorted', True)
        return undistorted

    def map_pixels_to_mm(self, points_px: np.ndarray) -> np.ndarray:
        """Transform pixel coordinates to physical mm using the calibration homography.
        
        The homography already encodes the full coordinate transform (including
        Y-axis orientation) because the calibration destination points are
        provided by the user in their physical coordinate system.
        """
        if self.homography_matrix is None:
            return points_px # No calibration
            
        # points_px should be shape (N, 2)
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix).reshape(-1, 2)
        return transformed

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

    def get_status(self) -> Dict:
        """Return the current calibration status and data."""
        if self.homography_matrix is None:
            return {"status": "uncalibrated"}
        return {
            "status": "calibrated",
            "matrix": self.homography_matrix.tolist(),
            "calibration_data": self.calibration_data
        }

    def check_calibration(self, image: np.ndarray) -> Dict:
        """
        Live calibration quality check.
        Detects visible AprilTags in the frame, maps their pixel centers through
        the homography, and compares to the stored physical positions.
        Returns per-tag reprojection error in mm and overall mean error.
        """
        if self.homography_matrix is None:
            return {"status": "uncalibrated", "error": "No calibration data found."}

        if self.calibration_data is None:
            return {
                "status": "calibrated",
                "warning": "Calibration matrix exists but no tag metadata was stored. "
                           "Re-run calibration to enable quality checking.",
                "per_tag_errors": [],
                "mean_error_mm": None,
            }

        detected = self.detect_tags(image)
        if not detected:
            return {
                "status": "calibrated",
                "warning": "No AprilTags visible in current frame.",
                "per_tag_errors": [],
                "mean_error_mm": None,
            }

        # Build physical position lookup from stored calibration_data
        phys_map = {p["id"]: p for p in self.calibration_data.get("physical_data", [])}

        per_tag_errors = []
        debug_img = self.undistort(image).copy()

        for tag in detected:
            tid = tag["id"]
            corners = np.array(tag["corners"])
            center_px = corners.mean(axis=0).astype(np.int32)
            
            # Draw on debug image
            cv2.polylines(debug_img, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(debug_img, f"ID:{tid}", (center_px[0], center_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if tid not in phys_map:
                continue

            p = phys_map[tid]
            center_px_f = corners.mean(axis=0).reshape(1, 2).astype(np.float32)
            center_mm = self.map_pixels_to_mm(center_px_f)[0]

            expected_mm = np.array([p["x"], p["y"]], dtype=np.float32)
            error_mm = float(np.linalg.norm(center_mm - expected_mm))

            per_tag_errors.append({
                "tag_id": tid,
                "detected_center_mm": center_mm.tolist(),
                "expected_mm": expected_mm.tolist(),
                "error_mm": round(error_mm, 2),
            })

        if not per_tag_errors:
            return {
                "status": "calibrated",
                "warning": "Detected tags do not match any stored calibration tags.",
                "per_tag_errors": [],
                "mean_error_mm": None,
            }

        mean_error = round(float(np.mean([t["error_mm"] for t in per_tag_errors])), 2)
        quality = "good" if mean_error < 5.0 else "poor"

        # Save debug image to disk
        debug_path = os.path.join(os.path.dirname(self.data_path), "calibration_debug.jpg")
        cv2.imwrite(debug_path, debug_img)
        
        return {
            "status": "calibrated",
            "quality": quality,
            "mean_error_mm": mean_error,
            "per_tag_errors": per_tag_errors,
            "debug_image_url": "/api/lens/calibration/debug-image"
        }

class AprilTagGenerator:
    @staticmethod
    def generate(tag_id: int, size_mm: float = 50.0, dpi: int = 300, return_pil: bool = False) -> Union[bytes, Image.Image]:
        """
        Generate an AprilTag image with a cut-guide border and a labeled footer.

        Layout (all measurements in "units", where 1 unit = size_px // 10):
        ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  ← cut-guide border (thin line)
          ┌───────────────────────┐
          │                       │      1-unit white gap on top and sides
          │   [  AprilTag  ]      │
          │                       │
          ├───────────────────────┤      bottom of tag
          │    ID: 0              │      text row (smaller, ~3mm)
          │                       │      1-unit gap below text
        └ └───────────────────────┘ ┘  ← cut-guide border + footer bottom border
          ↑                       ↑
          side borders extend through footer
        """
        # Core pixel size of the tag itself
        size_px = int((size_mm / 25.4) * dpi)

        # 1 "unit" = 10% of the tag size — used for all spacing
        unit = max(4, size_px // 10)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size_px)
        tag_img_rgb = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2RGB)

        # ── Text sizing ──────────────────────────────────────────────────────
        text = f"ID: {tag_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        border_thickness = max(1, unit // 8)

        # Target text height ~3mm (smaller than before)
        target_text_h_px = max(8, int((3.0 / 25.4) * dpi))
        font_scale = 1.0
        text_thickness = max(1, int(target_text_h_px / 22))
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        font_scale = target_text_h_px / float(th)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        # ── Canvas dimensions ────────────────────────────────────────────────
        # Top:    1 unit gap above tag
        # Middle: 1 unit gap below tag before divider line
        # Footer: text height + baseline + 1 unit gap below text
        top_gap    = unit
        tag_gap    = unit          # white space between tag bottom and divider
        text_row_h = th + baseline + unit
        footer_h   = tag_gap + text_row_h + unit

        total_w = size_px + 2 * unit
        total_h = size_px + top_gap + footer_h

        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # ── Paste tag ────────────────────────────────────────────────────────
        tag_y = top_gap
        tag_x = unit
        canvas[tag_y:tag_y + size_px, tag_x:tag_x + size_px] = tag_img_rgb

        # ── ID text — centered horizontally in footer ────────────────────────
        # Divider sits 1 unit below the tag bottom
        footer_top = tag_y + size_px + tag_gap
        text_x = (total_w - tw) // 2
        # Text baseline sits 1 unit below the divider + text height
        text_y = footer_top + unit + th
        cv2.putText(canvas, text, (text_x, text_y), font,
                    font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        # ── Cut-guide border ─────────────────────────────────────────────────
        cv2.rectangle(canvas,
                      (0, 0),
                      (total_w - 1, total_h - 1),
                      (0, 0, 0), border_thickness)

        # ── Footer side borders (extend from divider to bottom border) ───────
        cv2.line(canvas,
                 (0, footer_top),
                 (0, total_h - 1),
                 (0, 0, 0), border_thickness)
        cv2.line(canvas,
                 (total_w - 1, footer_top),
                 (total_w - 1, total_h - 1),
                 (0, 0, 0), border_thickness)

        # ── Horizontal divider between tag gap and footer text area ──────────
        cv2.line(canvas,
                 (0, footer_top),
                 (total_w - 1, footer_top),
                 (0, 0, 0), border_thickness)

        # ── Save ─────────────────────────────────────────────────────────────
        final_img = Image.fromarray(canvas)
        final_img.info["dpi"] = (dpi, dpi)

        if return_pil:
            return final_img

        buf = io.BytesIO()
        final_img.save(buf, format="PDF", resolution=dpi)
        return buf.getvalue()


    @staticmethod
    def generate_batch_document(start_id: int, count: int, size_mm: float = 50.0, dpi: int = 300, 
                                paper_width_in: float = 8.5, paper_height_in: float = 11.0) -> bytes:
        """
        Generate a multi-page PDF containing the requested tags packed densely.
        """
        paper_w_px = int(paper_width_in * dpi)
        paper_h_px = int(paper_height_in * dpi)
        
        # Get one tag to find its total size
        sample_tag = AprilTagGenerator.generate(start_id, size_mm, dpi, return_pil=True)
        tag_w, tag_h = sample_tag.size
        
        # Add padding between tags and page edges (10mm margin to prevent printer clipping)
        margin_mm = 10.0
        margin_px = int((margin_mm / 25.4) * dpi)
        
        usable_w = paper_w_px - 2 * margin_px
        usable_h = paper_h_px - 2 * margin_px
        
        tag_padding_mm = 5.0
        tag_padding_px = int((tag_padding_mm / 25.4) * dpi)
        
        cols = max(1, (usable_w + tag_padding_px) // (tag_w + tag_padding_px))
        rows = max(1, (usable_h + tag_padding_px) // (tag_h + tag_padding_px))
        tags_per_page = cols * rows
        
        pages = []
        current_page = None
        
        for i in range(count):
            tag_id = start_id + i
            tag_img = AprilTagGenerator.generate(tag_id, size_mm, dpi, return_pil=True)
            
            idx_on_page = i % tags_per_page
            
            if current_page is None or idx_on_page == 0:
                if current_page is not None:
                    pages.append(current_page)
                # Create a new white page
                current_page = Image.new("RGB", (paper_w_px, paper_h_px), (255, 255, 255))
                current_page.info["dpi"] = (dpi, dpi)
                
            row = idx_on_page // cols
            col = idx_on_page % cols
            
            x = margin_px + col * (tag_w + tag_padding_px)
            y = margin_px + row * (tag_h + tag_padding_px)
            
            current_page.paste(tag_img, (x, y))
            
        if current_page is not None:
            pages.append(current_page)
            
        if not pages:
            return b""
            
        # Save pages as PDF
        buf = io.BytesIO()
        if len(pages) == 1:
            pages[0].save(buf, format="PDF", resolution=dpi)
        else:
            pages[0].save(buf, format="PDF", resolution=dpi, save_all=True, append_images=pages[1:])
            
        return buf.getvalue()

# Global instance
calibration_service = CalibrationService()
