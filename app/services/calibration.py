import cv2
import numpy as np
import json
import os
import logging
import io
import time
import uuid
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple, Union

logger = logging.getLogger("vision_api.calibration")

class CalibrationService:
    def __init__(self, data_path: str = "calibration_data/matrix.json"):
        self.data_path = data_path
        self.lens_data_path = os.path.join(
            os.path.dirname(data_path), "lens_calibration.json"
        )
        self.homography_matrix = None
        self.calibration_data = None

        # Lens distortion model: standard pinhole or fisheye (Kannala-Brandt)
        self.is_fisheye = os.getenv("FISHEYE_LENS", "False").lower() == "true"

        # Lens intrinsics — loaded from lens_calibration.json if available
        self.camera_matrix = None   # K (3×3)
        self.dist_coeffs = None     # D (standard: 5 or 8 coeffs, fisheye: 4)
        self.lens_calibrated = False

        # Cached undistort remap tables (computed once, applied via cv2.remap)
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_resolution = None  # (w, h) the maps were built for

        # Y-axis flip: camera sees Y=0 at top, laser bed uses Y=0 at bottom-left.
        self.workspace_height_mm = float(os.getenv("WORKSPACE_HEIGHT_MM", "0"))

        self.load_calibration()
        self._load_lens_calibration()
        
        # AprilTag setup (using Aruco module)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # ── Tuning for small tags at high resolution ──────────────────────────
        # Default minMarkerPerimeterRate=0.03 requires perimeter >= 3% of image
        # longest side. At 4608px that's ~138px. A 20mm tag is ~100px, perimeter ~400px (0.08).
        # We can safely use 0.015 to ignore tiny noise but catch all tags.
        self.aruco_params.minMarkerPerimeterRate = 0.015
        self.aruco_params.maxMarkerPerimeterRate = 4.0

        # Widen the adaptive threshold window range significantly for 12MP.
        # A tag might be 100-150px wide. The window should be large enough to cover the tag.
        self.aruco_params.adaptiveThreshWinSizeMin = 13
        self.aruco_params.adaptiveThreshWinSizeMax = 153
        self.aruco_params.adaptiveThreshWinSizeStep = 10

        # Lower the threshold constant because HDR can sometimes flatten local contrast
        self.aruco_params.adaptiveThreshConstant = 5

        # 102 HFOV still has barrel distortion near edges (where bottom tags are).
        # Loosen the polygonal approximation to allow curved tag edges to be detected.
        self.aruco_params.polygonalApproxAccuracyRate = 0.06

        # Allow slightly more error in corner refinement for small/distorted tags
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementMaxIterations = 50
        self.aruco_params.cornerRefinementMinAccuracy = 0.05

        # Use more pixels per cell for bit extraction since we have higher resolution
        self.aruco_params.perspectiveRemovePixelPerCell = 8
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def load_calibration(self):
        """Load homography calibration data from JSON file."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.homography_matrix = np.array(data['matrix'])
                    self.calibration_data = data.get('calibration_data', None)
                logger.info(f"Loaded homography from {self.data_path}")
            except Exception as e:
                logger.error(f"Failed to load homography: {e}")

    def _load_lens_calibration(self):
        """Load lens intrinsics (K, D) from disk if available."""
        if not os.path.exists(self.lens_data_path):
            logger.info("No lens calibration found — undistortion disabled.")
            return
        try:
            with open(self.lens_data_path, 'r') as f:
                data = json.load(f)
            self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
            self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
            self.lens_calibrated = True
            stored_model = data.get('model', 'standard')
            logger.info(
                f"Loaded lens calibration (model={stored_model}, "
                f"RMS={data.get('rms_error', '?')}) from {self.lens_data_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load lens calibration: {e}")

    def _save_lens_calibration(
        self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
        rms_error: float, image_size: Tuple[int, int],
        num_images: int, model: str,
    ) -> None:
        """Persist lens intrinsics to disk."""
        os.makedirs(os.path.dirname(self.lens_data_path), exist_ok=True)
        data = {
            'camera_matrix': camera_matrix.tolist(),
            'dist_coeffs': dist_coeffs.tolist(),
            'rms_error': rms_error,
            'image_size': list(image_size),
            'num_images': num_images,
            'model': model,
        }
        with open(self.lens_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.lens_calibrated = True
        # Invalidate cached maps so they're rebuilt on next undistort call
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_resolution = None
        logger.info(
            f"Saved lens calibration (model={model}, RMS={rms_error:.4f}) "
            f"to {self.lens_data_path}"
        )

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

    def detect_tags(self, image: np.ndarray, apply_undistort: bool = True) -> List[Dict]:
        """
        Detect AprilTags in the image.

        Args:
            image: Input frame (BGR numpy array).
            apply_undistort: If True (default), applies fisheye undistortion
                before detection when FISHEYE_LENS is enabled. Pass False when
                the caller has already undistorted the frame to avoid processing
                the image twice.
        """
        img_processed = self.undistort(image) if apply_undistort else image

        corners, ids, rejected = self.detector.detectMarkers(img_processed)
        results = []
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                tag_corners = corners[i][0]  # 4x2 array of corners
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
        """Apply lens distortion correction using calibrated K/D coefficients.

        Uses cached remap tables for performance (~1ms per frame vs ~15ms for
        cv2.undistort). The maps are rebuilt automatically if the frame
        resolution changes.

        Returns the image unchanged if no lens calibration has been performed.
        """
        if not self.lens_calibrated:
            return image

        h, w = image.shape[:2]

        # Rebuild remap tables if resolution changed or first call
        if self._undistort_resolution != (w, h):
            logger.debug(f"Building undistort maps for {w}×{h}")
            if self.is_fisheye:
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix, self.dist_coeffs,
                    (w, h), np.eye(3), balance=1.0,
                )
                self._undistort_map1, self._undistort_map2 = \
                    cv2.fisheye.initUndistortRectifyMap(
                        self.camera_matrix, self.dist_coeffs,
                        np.eye(3), new_k, (w, h), cv2.CV_16SC2,
                    )
            else:
                new_k, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.dist_coeffs,
                    (w, h), alpha=1.0,
                )
                self._undistort_map1, self._undistort_map2 = \
                    cv2.initUndistortRectifyMap(
                        self.camera_matrix, self.dist_coeffs,
                        None, new_k, (w, h), cv2.CV_16SC2,
                    )
            self._undistort_resolution = (w, h)

        return cv2.remap(
            image, self._undistort_map1, self._undistort_map2,
            cv2.INTER_LINEAR,
        )

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

# ──────────────────────────────────────────────────────────────────────────────
# Checkerboard Generator
# ──────────────────────────────────────────────────────────────────────────────

class CheckerboardGenerator:
    """Generate printable checkerboard patterns for lens calibration."""

    @staticmethod
    def generate(
        rows: int = 9, cols: int = 6, square_mm: float = 25.0, dpi: int = 300,
    ) -> bytes:
        """
        Generate a printable PDF containing a checkerboard calibration pattern.

        Args:
            rows: Number of *inner* corners horizontally (squares = rows + 1).
            cols: Number of *inner* corners vertically (squares = cols + 1).
            square_mm: Physical side length of each square in mm.
            dpi: Print resolution.

        Returns:
            PDF file as bytes.
        """
        sq_px = int((square_mm / 25.4) * dpi)
        board_w = (rows + 1) * sq_px
        board_h = (cols + 1) * sq_px

        # Add margin for label text below the board
        margin = max(20, sq_px // 2)
        label_h = int((5.0 / 25.4) * dpi)  # ~5mm text height
        total_w = board_w + 2 * margin
        total_h = board_h + 2 * margin + label_h + margin

        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # Draw checkerboard
        for r in range(cols + 1):
            for c in range(rows + 1):
                if (r + c) % 2 == 0:
                    continue  # white square — already white
                x = margin + c * sq_px
                y = margin + r * sq_px
                cv2.rectangle(canvas, (x, y), (x + sq_px, y + sq_px), (0, 0, 0), -1)

        # Draw outer border
        cv2.rectangle(
            canvas, (margin, margin),
            (margin + board_w, margin + board_h), (0, 0, 0), 2,
        )

        # Label
        label = f"{rows}x{cols} inner corners | {square_mm:.1f}mm squares"
        font_scale = max(0.4, label_h / 30.0)
        text_thickness = max(1, int(font_scale))
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        text_x = (total_w - tw) // 2
        text_y = margin + board_h + margin + th
        cv2.putText(
            canvas, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness,
            cv2.LINE_AA,
        )

        img = Image.fromarray(canvas)
        img.info['dpi'] = (dpi, dpi)
        buf = io.BytesIO()
        img.save(buf, format='PDF', resolution=dpi)
        return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Lens Calibration Session
# ──────────────────────────────────────────────────────────────────────────────

# Zone names for the 3×3 placement grid
_ZONE_NAMES = [
    "top-left",    "top-center",    "top-right",
    "center-left", "center",        "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]

_ZONE_INSTRUCTIONS = {
    "top-left":       "Place the board in the TOP-LEFT area of the frame.",
    "top-center":     "Place the board in the TOP-CENTER area of the frame.",
    "top-right":      "Place the board in the TOP-RIGHT area of the frame.",
    "center-left":    "Place the board in the LEFT-CENTER area of the frame.",
    "center":         "Place the board in the CENTER of the frame.",
    "center-right":   "Place the board in the RIGHT-CENTER area of the frame.",
    "bottom-left":    "Place the board in the BOTTOM-LEFT area of the frame.",
    "bottom-center":  "Place the board in the BOTTOM-CENTER area of the frame.",
    "bottom-right":   "Place the board in the BOTTOM-RIGHT area of the frame.",
}


class LensCalibrationSession:
    """
    Manages a multi-step lens distortion calibration using a checkerboard.

    Usage:
        session = LensCalibrationSession(rows=9, cols=6, square_mm=25)
        session.start()
        while not session.can_finish:
            result = session.capture(frame)   # returns guidance dict
        report = session.finish(calibration_service)
    """

    MIN_CAPTURES = 10
    MAX_CAPTURES = 20
    MIN_ZONES    = 5

    def __init__(self, rows: int = 9, cols: int = 6, square_mm: float = 25.0):
        self.rows = rows
        self.cols = cols
        self.square_mm = square_mm
        self.session_id = str(uuid.uuid4())

        # Accumulated data
        self.obj_points_list: List[np.ndarray] = []   # 3-D world points
        self.img_points_list: List[np.ndarray] = []   # 2-D image points
        self.image_size: Optional[Tuple[int, int]] = None

        # Zone tracking
        self.zones_hit: Dict[str, int] = {}  # zone_name → capture count
        self.capture_count = 0
        self.last_preview: Optional[np.ndarray] = None

        # Pre-compute the 3-D object points for one checkerboard view
        self._obj_pts = np.zeros((rows * cols, 3), np.float32)
        self._obj_pts[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
        self._obj_pts *= square_mm  # scale to physical mm

        self.active = False

    def start(self) -> Dict:
        """Begin the calibration session."""
        self.active = True
        return self._status(instruction=self._next_instruction())

    def capture(self, frame: np.ndarray) -> Dict:
        """Attempt to detect the checkerboard in `frame`."""
        if not self.active:
            return {"error": "No active session. Call /start first."}

        if self.capture_count >= self.MAX_CAPTURES:
            return self._status(
                success=False,
                message="Maximum captures reached. Call /finish to compute calibration.",
            )

        h, w = frame.shape[:2]
        self.image_size = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), flags)

        if not found:
            self._save_preview(frame, None)
            return self._status(
                success=False,
                message="Checkerboard not detected. Ensure the full board is visible "
                        "and evenly lit. Avoid glare and shadows.",
            )

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Determine which zone the board center falls in
        center = corners.mean(axis=0).flatten()
        zone = self._classify_zone(center, w, h)

        # Store
        self.obj_points_list.append(self._obj_pts.copy())
        self.img_points_list.append(corners)
        self.capture_count += 1
        self.zones_hit[zone] = self.zones_hit.get(zone, 0) + 1

        self._save_preview(frame, corners)

        return self._status(
            success=True,
            zone_hit=zone,
            message=f"Capture #{self.capture_count} — board detected in '{zone}'.",
            instruction=self._next_instruction(),
        )

    @property
    def can_finish(self) -> bool:
        return (
            self.capture_count >= self.MIN_CAPTURES
            and len(self.zones_hit) >= self.MIN_ZONES
        )

    def finish(self, cal_service: 'CalibrationService') -> Dict:
        """Compute the lens calibration and save it."""
        if self.capture_count < 4:
            return {"error": "Need at least 4 captures to calibrate."}

        w, h = self.image_size
        is_fisheye = cal_service.is_fisheye

        if is_fisheye:
            # Kannala-Brandt fisheye model (4 coefficients)
            K = np.eye(3, dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_CHECK_COND
                | cv2.fisheye.CALIB_FIX_SKEW
            )
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6
            )
            # fisheye.calibrate expects obj_points as list of (N,1,3)
            obj_pts = [p.reshape(-1, 1, 3) for p in self.obj_points_list]
            img_pts = [p.reshape(-1, 1, 2) for p in self.img_points_list]

            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_pts, img_pts, (w, h), K, D,
                flags=flags, criteria=criteria,
            )
            D = D.flatten()
            model = "fisheye"
        else:
            # Standard pinhole + radial/tangential (5 coefficients)
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points_list, self.img_points_list, (w, h),
                None, None,
            )
            D = D.flatten()
            model = "standard"

        logger.info(f"Lens calibration complete: model={model}, RMS={rms:.4f}")

        cal_service._save_lens_calibration(
            camera_matrix=K,
            dist_coeffs=D,
            rms_error=float(rms),
            image_size=(w, h),
            num_images=self.capture_count,
            model=model,
        )

        self.active = False

        return {
            "status": "calibrated",
            "model": model,
            "rms_error": round(float(rms), 4),
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.tolist(),
            "captures_used": self.capture_count,
            "zones_covered": list(self.zones_hit.keys()),
            "note": (
                "Lens calibration saved. You should now re-run the AprilTag "
                "homography calibration (POST /api/lens/calibrate) since "
                "undistortion changes pixel positions."
            ),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _classify_zone(self, center: np.ndarray, w: int, h: int) -> str:
        """Map a point to one of the 9 grid zones."""
        cx, cy = center
        col = 0 if cx < w / 3 else (1 if cx < 2 * w / 3 else 2)
        row = 0 if cy < h / 3 else (1 if cy < 2 * h / 3 else 2)
        return _ZONE_NAMES[row * 3 + col]

    def _next_instruction(self) -> str:
        """Generate the next placement instruction based on zone coverage."""
        if self.capture_count == 0:
            return (
                "Hold the printed checkerboard flat in front of the camera. "
                "Start by placing it in the CENTER of the frame."
            )

        # Find uncovered zones
        uncovered = [z for z in _ZONE_NAMES if z not in self.zones_hit]
        if uncovered:
            target = uncovered[0]
            base = _ZONE_INSTRUCTIONS[target]
        else:
            # All zones covered — ask for a tilted view
            target = _ZONE_NAMES[self.capture_count % len(_ZONE_NAMES)]
            base = (
                f"All zones covered! For extra accuracy, tilt the board "
                f"~15° and place it in the {target.upper()} area."
            )

        remaining = max(0, self.MIN_CAPTURES - self.capture_count)
        if remaining > 0:
            base += f" ({remaining} more capture(s) needed)"
        elif self.can_finish:
            base += " (You can now call /finish, or keep adding captures for better accuracy.)"

        return base

    def _save_preview(self, frame: np.ndarray, corners: Optional[np.ndarray]) -> None:
        """Save a preview image with detected corners drawn."""
        preview = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(preview, (self.rows, self.cols), corners, True)
        # Draw 3×3 zone grid
        h, w = preview.shape[:2]
        for i in range(1, 3):
            cv2.line(preview, (w * i // 3, 0), (w * i // 3, h), (100, 100, 100), 1)
            cv2.line(preview, (0, h * i // 3), (w, h * i // 3), (100, 100, 100), 1)
        # Downscale for manageable file size
        MAX_W = 1280
        if w > MAX_W:
            s = MAX_W / w
            preview = cv2.resize(preview, (int(w * s), int(h * s)))
        self.last_preview = preview
        debug_path = "calibration_data/lens_calibration_preview.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, preview)

    def _status(self, **extra) -> Dict:
        """Build a status response dict."""
        return {
            "session_id": self.session_id,
            "captures_done": self.capture_count,
            "total_target": self.MIN_CAPTURES,
            "max_captures": self.MAX_CAPTURES,
            "zones_covered": list(self.zones_hit.keys()),
            "zones_remaining": [
                z for z in _ZONE_NAMES if z not in self.zones_hit
            ],
            "can_finish": self.can_finish,
            "preview_url": "/api/lens/calibrate-lens/preview",
            **extra,
        }


# ──────────────────────────────────────────────────────────────────────────────
# AprilTag Generator
# ──────────────────────────────────────────────────────────────────────────────

class AprilTagGenerator:
    @staticmethod
    def generate(tag_id: int, size_mm: float = 50.0, dpi: int = 300, return_pil: bool = False, guide_distance_mm: float = 0.0) -> Union[bytes, Image.Image]:
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

        if guide_distance_mm > 0.0:
            # guide_distance_mm = white margin from the tag edge to the cut-guide border.
            # e.g. 20mm tag with 5mm guide → border is 5mm outside the tag on all sides.
            guide_offset_px = max(1, int((guide_distance_mm / 25.4) * dpi))

            tag_x = guide_offset_px
            tag_y = guide_offset_px
            cut_w = size_px + 2 * guide_offset_px
            cut_h = size_px + 2 * guide_offset_px

            total_w = cut_w
            total_h = cut_h + footer_h  # text footer sits below the cut guide
        else:
            total_w = size_px + 2 * unit
            total_h = size_px + top_gap + footer_h
            cut_w = total_w
            cut_h = total_h
            
            tag_x = unit
            tag_y = top_gap
            center_x = tag_x + size_px // 2
            center_y = tag_y + size_px // 2

        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # ── Paste tag ────────────────────────────────────────────────────────
        paste_h = min(size_px, total_h - tag_y)
        paste_w = min(size_px, total_w - tag_x)
        if paste_h > 0 and paste_w > 0:
            canvas[tag_y:tag_y + paste_h, tag_x:tag_x + paste_w] = tag_img_rgb[:paste_h, :paste_w]

        # ── ID text — centered horizontally in footer ────────────────────────
        if guide_distance_mm > 0.0:
            footer_top = cut_h
        else:
            footer_top = tag_y + size_px + tag_gap

        text_x = (total_w - tw) // 2
        text_y = footer_top + unit + th
        if text_y < total_h:
            cv2.putText(canvas, text, (text_x, text_y), font,
                        font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        # ── Cut-guide border ─────────────────────────────────────────────────
        cv2.rectangle(canvas,
                      (0, 0),
                      (cut_w - 1, cut_h - 1),
                      (0, 0, 0), border_thickness)

        if guide_distance_mm <= 0.0:
            # ── Horizontal divider between tag gap and footer text area ──────────
            cv2.line(canvas,
                     (0, footer_top),
                     (total_w - 1, footer_top),
                     (0, 0, 0), border_thickness)
        else:
            # ── Box around the footer text ───────────────────────────────────────
            cv2.line(canvas, (0, cut_h), (0, total_h - 1), (0, 0, 0), border_thickness)
            cv2.line(canvas, (total_w - 1, cut_h), (total_w - 1, total_h - 1), (0, 0, 0), border_thickness)
            cv2.line(canvas, (0, total_h - 1), (total_w - 1, total_h - 1), (0, 0, 0), border_thickness)

            # ── Draw inward alignment ticks (8 ticks aligning with tag edges) ────
            tick_len_px = int((3.0 / 25.4) * dpi)
            
            # Top edge (pointing down)
            cv2.line(canvas, (tag_x, 0), (tag_x, tick_len_px), (0, 0, 0), border_thickness)
            cv2.line(canvas, (tag_x + size_px, 0), (tag_x + size_px, tick_len_px), (0, 0, 0), border_thickness)
            
            # Bottom edge (pointing up)
            cv2.line(canvas, (tag_x, cut_h - 1), (tag_x, cut_h - 1 - tick_len_px), (0, 0, 0), border_thickness)
            cv2.line(canvas, (tag_x + size_px, cut_h - 1), (tag_x + size_px, cut_h - 1 - tick_len_px), (0, 0, 0), border_thickness)
            
            # Left edge (pointing right)
            cv2.line(canvas, (0, tag_y), (tick_len_px, tag_y), (0, 0, 0), border_thickness)
            cv2.line(canvas, (0, tag_y + size_px), (tick_len_px, tag_y + size_px), (0, 0, 0), border_thickness)
            
            # Right edge (pointing left)
            cv2.line(canvas, (cut_w - 1, tag_y), (cut_w - 1 - tick_len_px, tag_y), (0, 0, 0), border_thickness)
            cv2.line(canvas, (cut_w - 1, tag_y + size_px), (cut_w - 1 - tick_len_px, tag_y + size_px), (0, 0, 0), border_thickness)

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
                                paper_width_in: float = 8.5, paper_height_in: float = 11.0, guide_distance_mm: float = 0.0) -> bytes:
        """
        Generate a multi-page PDF containing the requested tags packed densely.
        """
        paper_w_px = int(paper_width_in * dpi)
        paper_h_px = int(paper_height_in * dpi)
        
        # Get one tag to find its total size
        sample_tag = AprilTagGenerator.generate(start_id, size_mm, dpi, return_pil=True, guide_distance_mm=guide_distance_mm)
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
            tag_img = AprilTagGenerator.generate(tag_id, size_mm, dpi, return_pil=True, guide_distance_mm=guide_distance_mm)
            
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
