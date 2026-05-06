import os
import math
import time
import cv2
import logging
import numpy as np
from typing import List, Dict, Tuple

logger = logging.getLogger("vision_api.inference")


# ──────────────────────────────────────────────────────────────────────────────
# InferenceService
# ──────────────────────────────────────────────────────────────────────────────

class InferenceService:
    _REFERENCE_PATH = "calibration_data/reference_frame.jpg"

    def __init__(self):
        self.min_area_mm2       = float(os.getenv("MIN_WORKPIECE_AREA_MM2",   "500"))
        self.max_area_mm2       = float(os.getenv("MAX_WORKPIECE_AREA_MM2",   "160000"))
        self.grouping_reach     = int(os.getenv("DETECT_GROUPING_REACH_PX",   "25"))
        self.apriltag_mask_pad  = int(os.getenv("APRILTAG_MASK_PADDING",      "30"))
        self.change_threshold   = int(os.getenv("DETECT_CHANGE_THRESHOLD",    "25"))
        self.canny_high         = int(os.getenv("DETECT_CANNY_HIGH",          "60"))
        self.max_dim            = int(os.getenv("DETECT_MAX_DIM",             "1280"))
        
        # Cached reference frame (lazy-loaded at working resolution)
        self._ref_gray_proc: np.ndarray | None = None
        self._ref_sat_proc:  np.ndarray | None = None
        self._ref_proc_size: tuple[int, int] | None = None
        self.initialized = True
        logger.info("InferenceService initialised (modular pipeline).")

    # ── Reference Frame ────────────────────────────────────────────────────────

    @property
    def has_reference(self) -> bool:
        return os.path.exists(self._REFERENCE_PATH)

    def save_reference_frame(self, image: np.ndarray) -> None:
        """Save an empty-bed image as the background subtraction reference."""
        os.makedirs(os.path.dirname(self._REFERENCE_PATH), exist_ok=True)
        cv2.imwrite(self._REFERENCE_PATH, image)
        self._ref_gray_proc = None
        self._ref_sat_proc  = None
        self._ref_proc_size = None
        logger.info(f"Reference frame saved ({image.shape[1]}x{image.shape[0]}).")

    def clear_reference_frame(self) -> None:
        """Delete the reference frame (e.g. after lens recalibration)."""
        if os.path.exists(self._REFERENCE_PATH):
            os.remove(self._REFERENCE_PATH)
            logger.info("Reference frame deleted.")
        self._ref_gray_proc = None
        self._ref_sat_proc  = None
        self._ref_proc_size = None

    def _load_ref_channels(self, proc_w: int, proc_h: int) -> tuple:
        """Lazily load raw grayscale and HSV-saturation channels of the reference frame."""
        if not self.has_reference:
            return None, None
        if self._ref_gray_proc is not None and self._ref_proc_size == (proc_w, proc_h):
            return self._ref_gray_proc, self._ref_sat_proc
            
        raw = cv2.imread(self._REFERENCE_PATH)
        if raw is None:
            logger.warning("Reference frame unreadable.")
            return None, None
            
        resized = cv2.resize(raw, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        self._ref_gray_proc = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self._ref_sat_proc  = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)[:, :, 1]
        self._ref_proc_size = (proc_w, proc_h)
        logger.debug(f"Reference loaded at {proc_w}x{proc_h} (gray + saturation).")
        return self._ref_gray_proc, self._ref_sat_proc

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _odd(n: float) -> int:
        """Return the nearest odd integer ≥ 3 (OpenCV kernel requirement)."""
        n = max(3, int(n))
        return n if n % 2 == 1 else n + 1

    # ── Pipeline Steps ─────────────────────────────────────────────────────────

    def _get_working_resolution(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        orig_h, orig_w = image.shape[:2]
        long_side = max(orig_h, orig_w)
        if long_side > self.max_dim:
            ds = self.max_dim / long_side
            proc_w = int(orig_w * ds)
            proc_h = int(orig_h * ds)
            proc = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            ds = 1.0
            proc_w, proc_h = orig_w, orig_h
            proc = image
        return proc, ds, proc_w, proc_h

    def _compute_roi_mask(self, image: np.ndarray, ds: float, proc_w: int, proc_h: int, scale: float) -> np.ndarray:
        from .calibration import calibration_service
        tags_full = calibration_service.detect_tags(image, apply_undistort=False)
        tags_proc = [
            {**t, 'corners': (np.array(t['corners'], dtype=np.float32) * ds).tolist()}
            for t in tags_full
        ]
        
        mask_roi = np.zeros((proc_h, proc_w), dtype=np.uint8)
        if len(tags_proc) >= 3:
            pts = np.vstack([np.array(t['corners'], dtype=np.int32) for t in tags_proc])
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask_roi, [hull], 255)
        else:
            margin = int(80 * scale)
            mask_roi[margin:-margin, margin:-margin] = 255

        tag_pad = int(self.apriltag_mask_pad * scale)
        for tag in tags_proc:
            corners = np.array(tag['corners'], dtype=np.float32)
            centroid = corners.mean(axis=0)
            diag = np.linalg.norm(corners - centroid, axis=1).max()
            scale_f = 1.0 + tag_pad / max(diag, 1.0)
            expanded = centroid + (corners - centroid) * scale_f
            cv2.fillPoly(mask_roi, [expanded.astype(np.int32)], 0)

        if tag_pad > 0:
            erode_se = cv2.getStructuringElement(cv2.MORPH_RECT, (tag_pad * 2 + 1, tag_pad * 2 + 1))
            mask_roi = cv2.erode(mask_roi, erode_se, iterations=1)
            
        return mask_roi

    def _compute_brightness_diff(self, proc_gray: np.ndarray, ref_gray: np.ndarray, smooth_k: int) -> np.ndarray:
        gray_blurred = cv2.GaussianBlur(proc_gray, (smooth_k, smooth_k), 0)
        ref_blurred  = cv2.GaussianBlur(ref_gray, (smooth_k, smooth_k), 0)
        return cv2.absdiff(gray_blurred, ref_blurred)

    def _compute_saturation_diff(self, proc_hsv: np.ndarray, ref_sat: np.ndarray, smooth_k: int) -> np.ndarray:
        sat_live = proc_hsv[:, :, 1]
        sat_live_blur = cv2.GaussianBlur(sat_live, (smooth_k, smooth_k), 0)
        ref_sat_blur  = cv2.GaussianBlur(ref_sat, (smooth_k, smooth_k), 0)
        # Only positive saturation gain (0-255 clamped)
        return np.clip(
            sat_live_blur.astype(np.int16) - ref_sat_blur.astype(np.int16),
            0, 255
        ).astype(np.uint8)

    def _compute_combined_score(self, b_diff: np.ndarray, s_diff: np.ndarray) -> np.ndarray:
        # Simple addition clamped to 255
        return cv2.add(b_diff, s_diff)

    def _apply_threshold(self, combined_score: np.ndarray, threshold: int) -> np.ndarray:
        _, mask = cv2.threshold(combined_score, threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _apply_noise_filter(self, mask: np.ndarray, min_px: int = 15) -> np.ndarray:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) >= min_px:
                cv2.drawContours(clean, [c], -1, 255, -1)
        return clean

    def _group_and_fill_hulls(self, clean_mask: np.ndarray, morph_k: int, min_area: float) -> np.ndarray:
        group_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        group_mask = cv2.dilate(clean_mask, group_se)
        
        group_cnts, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(clean_mask)
        
        for g_cnt in group_cnts:
            if cv2.contourArea(g_cnt) < max(min_area * 0.05, 50):
                continue
                
            temp_g_mask = np.zeros_like(clean_mask)
            cv2.drawContours(temp_g_mask, [g_cnt], -1, 255, -1)
            
            object_pts_mask = cv2.bitwise_and(clean_mask, temp_g_mask)
            pts = np.argwhere(object_pts_mask > 0)
            
            if len(pts) > 5:
                pts_xy = pts[:, [1, 0]].astype(np.int32)
                hull = cv2.convexHull(pts_xy)
                cv2.fillPoly(hull_mask, [hull], 255)
                
        return hull_mask

    def _smooth_candidates(self, hull_mask: np.ndarray, mask_roi: np.ndarray) -> np.ndarray:
        # Decoupled small fixed kernel for closing (e.g. 5x5) to prevent boundary inflation
        close_se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        candidate_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, close_se)
        return cv2.bitwise_and(candidate_mask, candidate_mask, mask=mask_roi)

    def _extract_results(self, candidate_mask: np.ndarray, ds: float, proc_w: int, proc_h: int, min_area: float) -> List[Dict]:
        final_cnts, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        from .calibration import calibration_service as cal

        for i, cnt in enumerate(final_cnts):
            if cv2.contourArea(cnt) < min_area:
                continue
            if cv2.contourArea(cnt) > proc_w * proc_h * 0.5:
                continue

            cnt_full = (cnt.astype(np.float64) / ds).astype(np.int32) if ds < 1.0 else cnt

            rect = cv2.minAreaRect(cnt_full)
            corners_px = cv2.boxPoints(rect).astype(np.float32)

            edges = [corners_px[1] - corners_px[0], corners_px[2] - corners_px[1]]
            long_e = max(edges, key=lambda e: float(e[0]**2 + e[1]**2))
            angle_deg = math.degrees(math.atan2(float(long_e[1]), float(long_e[0])))
            while angle_deg > 45.0: angle_deg -= 90.0
            while angle_deg < -45.0: angle_deg += 90.0

            corners_px = corners_px.astype(np.int32)
            hull_full = cv2.convexHull(cnt_full)
            seg_px = hull_full.reshape(-1, 2).tolist()
            x, y, w, h = cv2.boundingRect(cnt_full)

            corners_mm = cal.map_pixels_to_mm(corners_px.astype(np.float32))
            seg_mm = cal.map_pixels_to_mm(np.array(seg_px, dtype=np.float32))

            area_mm2 = float(cv2.contourArea(corners_mm.reshape(-1, 1, 2).astype(np.float32)))
            
            if not (self.min_area_mm2 <= area_mm2 <= self.max_area_mm2):
                continue

            box_mm = [
                float(corners_mm[:, 0].min()),
                float(corners_mm[:, 1].min()),
                float(corners_mm[:, 0].max()),
                float(corners_mm[:, 1].max()),
            ]

            results.append({
                "id":               f"wp_{len(results):03d}",
                "class":            "workpiece",
                "confidence":       1.0,
                "angle_deg":        round(angle_deg, 2),
                "box_px":           [x, y, x + w, y + h],
                "corners_px":       corners_px.tolist(),
                "segmentation_px":  seg_px,
                "corners_mm":       corners_mm.tolist(),
                "segmentation_mm":  seg_mm.tolist(),
                "box_mm":           box_mm,
                "area_mm2":         round(area_mm2, 1),
            })
            
        return results

    # ── Core Detect ────────────────────────────────────────────────────────────

    def _detect_objects(self, image: np.ndarray):
        t0 = time.monotonic()
        
        # 1. Working Resolution
        proc_image, ds, proc_w, proc_h = self._get_working_resolution(image)
        scale = min(proc_h, proc_w) / 720.0
        smooth_k = self._odd(11 * scale)
        morph_k  = self._odd(self.grouping_reach * scale)
        min_area = max(100, int(1000 * scale * scale))

        # 2. ROI Mask
        mask_roi = self._compute_roi_mask(image, ds, proc_w, proc_h, scale)
        
        proc_gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)
        proc_hsv  = cv2.cvtColor(proc_image, cv2.COLOR_BGR2HSV)
        
        ref_gray, ref_sat = self._load_ref_channels(proc_w, proc_h)

        b_diff = np.zeros((proc_h, proc_w), dtype=np.uint8)
        s_diff = np.zeros((proc_h, proc_w), dtype=np.uint8)
        combined_score = np.zeros((proc_h, proc_w), dtype=np.uint8)

        if ref_gray is not None:
            # Subtraction path
            b_diff = self._compute_brightness_diff(proc_gray, ref_gray, smooth_k)
            s_diff = self._compute_saturation_diff(proc_hsv, ref_sat, smooth_k)
            combined_score = self._compute_combined_score(b_diff, s_diff)
            raw_mask = self._apply_threshold(combined_score, self.change_threshold)
        else:
            # Otsu fallback (no reference)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(proc_gray)
            blurred = cv2.GaussianBlur(enhanced, (smooth_k, smooth_k), 0)
            roi_pixels = blurred[mask_roi > 0]
            otsu_t = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0] if len(roi_pixels) > 0 else 127
            raw_mask = cv2.threshold(blurred, otsu_t, 255, cv2.THRESH_BINARY)[1]

        raw_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=mask_roi)

        # Pipeline Steps
        clean_mask     = self._apply_noise_filter(raw_mask)
        hull_mask      = self._group_and_fill_hulls(clean_mask, morph_k, min_area)
        candidate_mask = self._smooth_candidates(hull_mask, mask_roi)
        results        = self._extract_results(candidate_mask, ds, proc_w, proc_h, min_area)

        elapsed = time.monotonic() - t0
        logger.info(f"Detection: {len(results)} workpiece(s) in {elapsed:.3f}s")
        
        # Return all step outputs for the debug image generator
        return results, b_diff, s_diff, combined_score, raw_mask, clean_mask, hull_mask, candidate_mask

    # ── Debug Composite ────────────────────────────────────────────────────────

    def _save_debug_image(self, image: np.ndarray, results: List[Dict], steps_data: tuple) -> None:
        """
        Write a 3x3 9-panel grid to calibration_data/detect_debug.jpg:
          [A] Original       [B] Brightness Diff   [C] Saturation Diff
          [D] Combined       [E] Binary Threshold  [F] Noise Filtered
          [G] Hull Grouping  [H] Morphological     [I] Final Detections
        """
        b_diff, s_diff, combined_score, raw_mask, clean_mask, hull_mask, candidate_mask = steps_data
        
        debug_path = "calibration_data/detect_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        try:
            MAX_W = 1280
            orig_h, orig_w = image.shape[:2]
            s = min(1.0, MAX_W / orig_w)
            tw, th = int(orig_w * s), int(orig_h * s)

            def _resize(img): return cv2.resize(img, (tw, th))
            def _to_bgr(mask): return cv2.cvtColor(_resize(mask), cv2.COLOR_GRAY2BGR)

            # Panel A: Original Undistorted
            panel_a = _resize(image)

            # Panel B: Brightness Diff (HOT)
            panel_b = cv2.applyColorMap(_resize(b_diff), cv2.COLORMAP_HOT)

            # Panel C: Saturation Diff (COOL)
            panel_c = cv2.applyColorMap(_resize(s_diff), cv2.COLORMAP_COOL)

            # Panel D: Combined Score (JET)
            panel_d = cv2.applyColorMap(_resize(combined_score), cv2.COLORMAP_JET)

            # Panel E: Binary Threshold
            panel_e = _to_bgr(raw_mask)

            # Panel F: After Noise Filter
            panel_f = _to_bgr(clean_mask)

            # Panel G: After Hull Grouping
            panel_g = _to_bgr(hull_mask)

            # Panel H: After Morphological Close
            panel_h = _to_bgr(candidate_mask)

            # Panel I: Final Detections (Original + Overlays)
            overlay = panel_a.copy()
            for wp in results:
                pts = (np.array(wp['corners_px'], dtype=np.float32) * s).astype(np.int32)
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                angle_r = math.radians(wp.get('angle_deg', 0))
                ax = int(cx + 40 * math.cos(angle_r))
                ay = int(cy + 40 * math.sin(angle_r))

                cv2.polylines(overlay, [pts], True, (0, 220, 0), 2)
                cv2.arrowedLine(overlay, (cx, cy), (ax, ay), (0, 140, 255), 2, tipLength=0.25)
            panel_i = overlay

            lbl_args = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.putText(panel_a, "A: Original", (8, 24), **lbl_args)
            cv2.putText(panel_b, "B: Brightness Diff", (8, 24), **lbl_args)
            cv2.putText(panel_c, "C: Saturation Diff", (8, 24), **lbl_args)
            cv2.putText(panel_d, "D: Combined Score", (8, 24), **lbl_args)
            cv2.putText(panel_e, "E: Binary Threshold", (8, 24), **lbl_args)
            cv2.putText(panel_f, "F: Noise Filtered", (8, 24), **lbl_args)
            cv2.putText(panel_g, "G: Hull Grouping", (8, 24), **lbl_args)
            cv2.putText(panel_h, "H: Morphological", (8, 24), **lbl_args)
            cv2.putText(panel_i, "I: Final Detections", (8, 24), **lbl_args)

            row1 = np.hstack([panel_a, panel_b, panel_c])
            row2 = np.hstack([panel_d, panel_e, panel_f])
            row3 = np.hstack([panel_g, panel_h, panel_i])
            cv2.imwrite(debug_path, np.vstack([row1, row2, row3]))

        except Exception as exc:
            logger.warning(f"Debug composite failed: {exc}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_workpieces(self, image: np.ndarray) -> List[Dict]:
        if image is None:
            return []

        from .calibration import calibration_service
        image = calibration_service.undistort(image)

        mean_brightness = float(np.mean(image))
        if mean_brightness < 15:
            logger.warning(f"Frame very dark (mean={mean_brightness:.1f}) — AE may still be settling.")

        results, *steps_data = self._detect_objects(image)
        self._save_debug_image(image, results, steps_data)
        
        return results


# Singleton
inference_service = InferenceService()
