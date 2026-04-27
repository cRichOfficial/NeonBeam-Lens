import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger("vision_api.transform")

class TransformService:
    def calculate_transform(self, design_data: Dict[str, Any], workpiece_data: Dict[str, Any], padding_mm: float = 5.0) -> Dict[str, Any]:
        """
        Calculate transformation parameters to fit a design onto a workpiece.
        design_data: { "width_mm": float, "height_mm": float } OR { "image_base64": str, "dpi": float }
        workpiece_data: { "corners_mm": [[x,y], [x,y], [x,y], [x,y]] }
        """
        # 1. Get Design Dimensions
        d_w, d_h = self._get_design_dims(design_data)
        
        # 2. Get Workpiece Properties
        wp_corners = np.array(workpiece_data["corners_mm"])
        wp_center = wp_corners.mean(axis=0)
        
        # Calculate workpiece orientation and size
        # We'll assume the workpiece is roughly rectangular
        # Find the vector of the first side (TL to TR)
        v1 = wp_corners[1] - wp_corners[0]
        wp_angle = np.degrees(np.arctan2(v1[1], v1[0]))
        
        # Calculate widths and heights along the oriented axes
        # (Simplified: just distance between corners)
        wp_w = np.linalg.norm(v1)
        wp_h = np.linalg.norm(wp_corners[2] - wp_corners[1])
        
        logger.debug(f"Design: {d_w}x{d_h}mm, Workpiece: {wp_w:.1f}x{wp_h:.1f}mm at {wp_angle:.1f} deg")

        # 3. Auto-Rotate (if aspect ratios don't match)
        # If design is landscape and workpiece is portrait (or vice versa), rotate 90
        needs_90_rot = False
        if (d_w > d_h and wp_h > wp_w) or (d_h > d_w and wp_w > wp_h):
            needs_90_rot = True
            d_w, d_h = d_h, d_w
            
        # 4. Auto-Scale
        # Fit within workpiece with padding
        available_w = max(0, wp_w - 2 * padding_mm)
        available_h = max(0, wp_h - 2 * padding_mm)
        
        scale_w = available_w / d_w if d_w > 0 else 1.0
        scale_h = available_h / d_h if d_h > 0 else 1.0
        scale = min(scale_w, scale_h)
        
        # 5. Final Parameters
        final_rotation = wp_angle + (90 if needs_90_rot else 0)
        
        return {
            "rotation_deg": float(final_rotation),
            "scale": float(scale),
            "translation_mm": {
                "x": float(wp_center[0]),
                "y": float(wp_center[1])
            },
            "workpiece_center_mm": wp_center.tolist(),
            "design_original_mm": {"w": d_w, "h": d_h}
        }

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
