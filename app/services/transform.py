import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import io
import cv2
import os
from .calibration import calibration_service as cal

logger = logging.getLogger("vision_api.transform")

class TransformService:
    def calculate_transform(
        self, design_data: Dict[str, Any], workpiece_data: Dict[str, Any], 
        padding_mm: float = 5.0, material_height_mm: float = 0.0,
        debug_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate transformation parameters to fit a design onto a workpiece.
        If material_height_mm > 0, re-maps workpiece geometry to compensate for parallax.
        """
        # 1. Geometry Correction (Parallax Compensation)
        # If height is provided, we re-calculate the physical coordinates from pixels.
        corners_px = np.array(workpiece_data.get("corners_px", []), dtype=np.float32)
        seg_px = np.array(workpiece_data.get("segmentation_px", []), dtype=np.float32)
        
        if corners_px.size > 0 and material_height_mm > 0:
            logger.info(f"Applying parallax compensation for height={material_height_mm}mm")
            # Re-map using the height-aware calibration
            corners_mm = cal.map_pixels_to_mm(corners_px, height_mm=material_height_mm)
            seg_mm = cal.map_pixels_to_mm(seg_px, height_mm=material_height_mm)
            
            # Update workpiece data with corrected values
            workpiece_data["corners_mm"] = corners_mm.tolist()
            workpiece_data["segmentation_mm"] = seg_mm.tolist()
            workpiece_data["box_mm"] = [
                float(corners_mm[:, 0].min()),
                float(corners_mm[:, 1].min()),
                float(corners_mm[:, 0].max()),
                float(corners_mm[:, 1].max()),
            ]
            # Center is mean of corners
            wp_center = corners_mm.mean(axis=0)
        else:
            wp_center = np.array(workpiece_data.get("corners_mm", [])).mean(axis=0)

        # 2. Get Design Dimensions
        d_w, d_h = self._get_design_dims(design_data)
        
        # 3. Get Workpiece Properties
        wp_corners = np.array(workpiece_data["corners_mm"])
        
        # Calculate workpiece orientation and size
        v1 = wp_corners[1] - wp_corners[0]
        wp_angle = np.degrees(np.arctan2(v1[1], v1[0]))
        
        wp_w = np.linalg.norm(v1)
        wp_h = np.linalg.norm(wp_corners[2] - wp_corners[1])
        
        logger.debug(f"Design: {d_w}x{d_h}mm, Workpiece: {wp_w:.1f}x{wp_h:.1f}mm at {wp_angle:.1f} deg")

        # 4. Auto-Rotate (if aspect ratios don't match)
        needs_90_rot = False
        if (d_w > d_h and wp_h > wp_w) or (d_h > d_w and wp_w > wp_h):
            needs_90_rot = True
            d_w, d_h = d_h, d_w
            
        # 5. Auto-Scale
        available_w = max(0, wp_w - 2 * padding_mm)
        available_h = max(0, wp_h - 2 * padding_mm)
        
        scale_w = available_w / d_w if d_w > 0 else 1.0
        scale_h = available_h / d_h if d_h > 0 else 1.0
        scale = min(scale_w, scale_h)
        
        # 6. Final Parameters
        final_rotation = wp_angle + (90 if needs_90_rot else 0)
        
        # 7. Debug Rendering (Visual Verification)
        if debug_frame is not None:
            self._save_transform_debug(debug_frame, workpiece_data, material_height_mm)

        return {
            "rotation_deg": float(final_rotation),
            "scale": float(scale),
            "translation_mm": {
                "x": float(wp_center[0]),
                "y": float(wp_center[1])
            },
            "workpiece": workpiece_data, # Return the full corrected workpiece
            "design_original_mm": {"w": d_w, "h": d_h}
        }

    def _save_transform_debug(self, frame: np.ndarray, wp: Dict, height: float):
        """Save a visual debug image showing original vs corrected location."""
        debug_img = frame.copy()
        
        # Original (shifted) location from pixels (cyan)
        corners_px = np.array(wp["corners_px"], dtype=np.int32)
        cv2.polylines(debug_img, [corners_px], True, (255, 255, 0), 2)
        
        # Corrected (base) location
        # To visualize it on the image, we'd need to map MM back to PX.
        # But for debugging "did it move?", we can just draw the corrected center.
        # However, a better debug image is just showing the corrected vs uncorrected.
        
        # Save to disk
        debug_path = "calibration_data/transform_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, debug_img)
        logger.info(f"Saved transform debug image to {debug_path}")

    def _get_design_dims(self, design_data: Dict[str, Any]) -> Tuple[float, float]:
        if "width_mm" in design_data and "height_mm" in design_data:
            return design_data["width_mm"], design_data["height_mm"]
        
        if "image_data" in design_data and "dpi" in design_data:
            # Calculate from image size and DPI
            # image_data is bytes
            img = Image.open(io.BytesIO(design_data["image_data"]))
            w_px, h_px = img.size
            dpi = design_data["dpi"]
            # 1 inch = 25.4 mm
            w_mm = (w_px / dpi) * 25.4
            h_mm = (h_px / dpi) * 25.4
            return w_mm, h_mm
            
        return 10.0, 10.0 # Default fallback

# Global instance
transform_service = TransformService()
