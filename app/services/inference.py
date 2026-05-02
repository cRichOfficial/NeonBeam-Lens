import os
import time
import cv2
import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger("vision_api.inference")


# ──────────────────────────────────────────────────────────────────────────────
# Detection Service
# ──────────────────────────────────────────────────────────────────────────────

class InferenceService:
    """
    OpenCV-based workpiece detector for a black honeycomb laser bed.

    Detection strategy: **Local texture variance.**
    The honeycomb bed has a distinctive repeating hole pattern which produces
    high local pixel variance.  Any solid workpiece (wood, slate, acrylic,
    card, metal) placed on it has significantly *lower* local variance because
    it is a continuous surface.  By thresholding on low-variance regions inside
    the AprilTag-bounded ROI, we detect objects of *any* brightness — bright
    wood and dark slate alike.

    All expensive processing runs on a downsampled working image (capped at
    DETECT_MAX_DIM px on the long side).  Contour coordinates are scaled back
    to full resolution before the calibration homography maps them to physical
    mm values.
    """

    def __init__(self):
        self.min_area_mm2 = float(os.getenv("MIN_WORKPIECE_AREA_MM2", "500"))
        self.max_area_mm2 = float(os.getenv("MAX_WORKPIECE_AREA_MM2", "160000"))
        self.honeycomb_kernel = int(os.getenv("HONEYCOMB_KERNEL_SIZE", "25"))
        self.apriltag_mask_padding = int(os.getenv("APRILTAG_MASK_PADDING", "30"))
        self.variance_window = int(os.getenv("DETECT_VARIANCE_WINDOW", "25"))
        # Variance threshold controls — tune via debug image Panel C
        self.variance_percentile = float(os.getenv("DETECT_VARIANCE_PERCENTILE", "70"))
        self.variance_multiplier = float(os.getenv("DETECT_VARIANCE_MULTIPLIER", "0.25"))
        self.initialized = True
        logger.info("InferenceService initialised (OpenCV texture-variance detector).")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _odd(n: int) -> int:
        """Round to the nearest odd integer ≥ 3 (required for OpenCV kernels)."""
        n = max(3, int(n))
        return n if n % 2 == 1 else n + 1

    # ── Core Detection ─────────────────────────────────────────────────────────

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect workpieces on a honeycomb laser bed using local texture variance.

        Returns a list of workpiece dicts (see `detect_workpieces` for schema).
        `image` must already be undistorted if FISHEYE_LENS is enabled.
        """
        orig_h, orig_w = image.shape[:2]
        t0 = time.monotonic()

        # ── 1. Downsample to capped working resolution ───────────────────────
        max_dim = int(os.getenv("DETECT_MAX_DIM", "1280"))
        long_side = max(orig_h, orig_w)
        if long_side > max_dim:
            ds = max_dim / long_side
            proc_w = int(orig_w * ds)
            proc_h = int(orig_h * ds)
            proc = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            ds = 1.0
            proc_w, proc_h = orig_w, orig_h
            proc = image

        logger.debug(
            f"Working res: {proc_w}×{proc_h} "
            f"(ds={ds:.3f} from {orig_w}×{orig_h})"
        )

        # ── 2. Preprocess ────────────────────────────────────────────────────
        ref_dim = 720.0
        scale = min(proc_h, proc_w) / ref_dim

        smooth_k = self._odd(11 * scale)
        morph_k  = self._odd(self.honeycomb_kernel * scale)
        var_k    = self._odd(self.variance_window * scale)
        tag_pad  = int(self.apriltag_mask_padding * scale)
        min_area = max(100, int(1000 * scale * scale))

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        smoothed = cv2.GaussianBlur(enhanced, (smooth_k, smooth_k), 0)

        # ── 3. Build ROI mask from AprilTag convex hull ──────────────────────
        #    Detect tags on the FULL-res image (ArUco params are tuned for it),
        #    then scale corner coordinates to proc-space.
        from .calibration import calibration_service
        tags_full = calibration_service.detect_tags(image, apply_undistort=False)
        logger.debug(f"AprilTags detected (full-res): {len(tags_full)}")

        tags_proc = []
        for t in tags_full:
            scaled = (np.array(t['corners'], dtype=np.float32) * ds).tolist()
            tags_proc.append({**t, 'corners': scaled})

        mask_roi = np.zeros((proc_h, proc_w), dtype=np.uint8)
        if len(tags_proc) >= 3:
            pts = np.vstack([np.array(t['corners'], dtype=np.int32) for t in tags_proc])
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask_roi, [hull], 255)
            roi_px = cv2.countNonZero(mask_roi)
            logger.debug(
                f"ROI: {len(tags_proc)}-tag hull, "
                f"{roi_px} px ({100 * roi_px / (proc_w * proc_h):.1f}% of frame)"
            )
        else:
            margin = int(80 * scale)
            mask_roi[margin:-margin, margin:-margin] = 255
            logger.warning(
                f"Only {len(tags_proc)} tags detected — "
                f"using margin fallback ROI (margin={margin}px)"
            )

        # Erase AprilTag squares + white paper backing from ROI
        for tag in tags_proc:
            corners = np.array(tag['corners'], dtype=np.float32)
            centroid = corners.mean(axis=0)
            half_diag = np.linalg.norm(corners - centroid, axis=1).max()
            expand = 1.0 + tag_pad / max(half_diag, 1.0)
            expanded = centroid + (corners - centroid) * expand
            cv2.fillPoly(mask_roi, [expanded.astype(np.int32)], 0)

        # Erode ROI inward by tag_pad to avoid using pixels right at the hull
        # boundary (they produce artificially low variance artefacts).
        if tag_pad > 0:
            erode_se = cv2.getStructuringElement(
                cv2.MORPH_RECT, (tag_pad * 2 + 1, tag_pad * 2 + 1)
            )
            mask_roi = cv2.erode(mask_roi, erode_se, iterations=1)

        # ── 4. Local variance — texture-based detection ──────────────────────
        #    Variance = E[X²] − E[X]²   (two box filters, very fast)
        #
        #    Honeycomb → high variance (repeating hole pattern)
        #    Solid workpiece → low variance (continuous surface)
        sf = smoothed.astype(np.float32)
        mean_img = cv2.blur(sf, (var_k, var_k))
        mean_sq  = cv2.blur(sf * sf, (var_k, var_k))
        local_var = np.maximum(mean_sq - mean_img * mean_img, 0.0)

        # ── 5. Threshold: low-variance regions inside ROI ────────────────────
        roi_var = local_var[mask_roi > 0]
        if len(roi_var) == 0:
            logger.warning("ROI is empty — cannot detect objects.")
            return []

        # Use configurable percentile + multiplier — tune via debug Panel C.
        # A solid workpiece is typically <25% of the honeycomb variance.
        hc_baseline = float(np.percentile(roi_var, self.variance_percentile))
        var_thresh   = hc_baseline * self.variance_multiplier

        logger.debug(
            f"Variance: HC baseline (p60)={hc_baseline:.1f}, "
            f"threshold={var_thresh:.1f}, "
            f"ROI var min={float(roi_var.min()):.1f}, "
            f"ROI var max={float(roi_var.max()):.1f}"
        )

        # Binary mask: low-variance pixels inside the ROI
        low_var = (local_var < var_thresh).astype(np.uint8) * 255
        workpiece_mask = cv2.bitwise_and(low_var, low_var, mask=mask_roi)
        # Keep a copy of the raw threshold mask for the debug panel
        raw_var_mask = workpiece_mask.copy()

        # ── 6. Two-pass morphology ───────────────────────────────────────────
        #   Pass A ─ small close: fills honeycomb holes *within* a workpiece patch
        morph_se = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        closed = cv2.morphologyEx(workpiece_mask, cv2.MORPH_CLOSE, morph_se)

        #   Pass B ─ large close: merges fragments of the same workpiece
        #   (kernel = 3× the honeycomb spacing to bridge gaps across the piece)
        large_k = self._odd(morph_k * 3)
        large_se = cv2.getStructuringElement(cv2.MORPH_RECT, (large_k, large_k))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, large_se)

        #   Erode boundary artefacts (thin fringe left by the ROI edge),
        #   then dilate back to restore true workpiece size.
        boundary_se = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        eroded  = cv2.erode(closed, boundary_se, iterations=1)
        dilated = cv2.dilate(eroded, boundary_se, iterations=2)

        # ── 7. Find contours (proc-space) ────────────────────────────────────
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.debug(f"Contours before area filter: {len(contours)}")

        # ── 8. Build results — scale contours to full-res ────────────────────
        results = []
        from .calibration import calibration_service as cal

        for i, cnt in enumerate(contours):
            area_proc = cv2.contourArea(cnt)
            if area_proc < min_area or area_proc > (proc_w * proc_h * 0.5):
                continue

            # Scale contour from proc-space → full-res
            if ds < 1.0:
                cnt_full = (cnt.astype(np.float64) / ds).astype(np.int32)
            else:
                cnt_full = cnt

            # Oriented bounding rect → 4 corners + angle
            rect = cv2.minAreaRect(cnt_full)
            corners_px = cv2.boxPoints(rect).astype(np.int32)
            angle_deg = float(rect[2])

            # Convex hull as segmentation polygon
            hull = cv2.convexHull(cnt_full)
            seg_px = hull.reshape(-1, 2).tolist()

            # Axis-aligned bounding box (full-res pixels)
            x, y, w, h = cv2.boundingRect(cnt_full)

            # Map to physical mm via calibration homography
            corners_mm = cal.map_pixels_to_mm(corners_px.astype(np.float32))
            seg_mm = cal.map_pixels_to_mm(
                np.array(seg_px, dtype=np.float32)
            )

            # Physical area filter
            area_mm2 = float(
                cv2.contourArea(corners_mm.reshape(-1, 1, 2).astype(np.float32))
            )
            if area_mm2 < self.min_area_mm2 or area_mm2 > self.max_area_mm2:
                logger.debug(
                    f"Filtered detection {i}: area {area_mm2:.0f} mm² "
                    f"out of [{self.min_area_mm2}, {self.max_area_mm2}]"
                )
                continue

            box_mm = [
                float(corners_mm[:, 0].min()),
                float(corners_mm[:, 1].min()),
                float(corners_mm[:, 0].max()),
                float(corners_mm[:, 1].max()),
            ]

            results.append({
                "id": f"wp_{len(results):03d}",
                "class": "workpiece",
                "confidence": 1.0,
                "angle_deg": angle_deg,
                "box_px": [x, y, x + w, y + h],
                "corners_px": corners_px.tolist(),
                "segmentation_px": seg_px,
                "corners_mm": corners_mm.tolist(),
                "segmentation_mm": seg_mm.tolist(),
                "box_mm": box_mm,
                "area_mm2": area_mm2,
            })

        elapsed = time.monotonic() - t0
        logger.info(
            f"Detection complete: {len(results)} workpiece(s) in {elapsed:.2f}s"
        )
        return results, raw_var_mask, proc_w, proc_h

    # ── Debug Image ────────────────────────────────────────────────────────────

    def _save_debug_image(
        self, image: np.ndarray, results: List[Dict],
        mean_brightness: float,
        var_mask_proc: np.ndarray | None = None,
        proc_w: int = 0, proc_h: int = 0,
    ) -> None:
        """
        Save a 3-panel diagnostic composite:
          Panel A (left)   — raw frame with detection overlays
          Panel B (centre) — CLAHE-enhanced grayscale (detector input)
          Panel C (right)  — raw variance threshold mask (before morphology)
        """
        debug_path = "calibration_data/detect_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        debug_img = image.copy()

        # Draw detection overlays
        for wp in results:
            pts = np.array(wp['corners_px'], dtype=np.int32)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                debug_img, wp['id'],
                (pts[0][0], pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            for px, mm in zip(pts, wp['corners_mm']):
                label = f"({int(mm[0])},{int(mm[1])})"
                cv2.putText(
                    debug_img, label, (px[0], px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )

        try:
            MAX_W = 1280
            h, w = debug_img.shape[:2]
            s = min(1.0, MAX_W / w)
            tw, th = int(w * s), int(h * s)

            panel_a = cv2.resize(debug_img, (tw, th))

            gray = cv2.cvtColor(cv2.resize(image, (tw, th)), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            panel_b = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

            # Panel C: variance mask upscaled to match panel size
            if var_mask_proc is not None and var_mask_proc.size > 0:
                panel_c = cv2.cvtColor(
                    cv2.resize(var_mask_proc, (tw, th), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR
                )
                # Tint low-variance (white) pixels cyan so workpiece pops
                cyan_tint = np.zeros_like(panel_c)
                cyan_tint[panel_c[:, :, 0] > 128] = [255, 255, 0]  # cyan = B+G
                panel_c = cv2.addWeighted(panel_c, 0.4, cyan_tint, 0.6, 0)
            else:
                panel_c = np.zeros((th, tw, 3), dtype=np.uint8)

            lbl = dict(
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color=(0, 255, 255), thickness=2,
            )
            cv2.putText(panel_a, f"RAW  mean={mean_brightness:.0f}", (10, 30), **lbl)
            cv2.putText(panel_b, "CLAHE gray (detector input)", (10, 30), **lbl)
            cv2.putText(panel_c, "Variance mask (pre-morphology)", (10, 30), **lbl)

            cv2.imwrite(debug_path, np.hstack([panel_a, panel_b, panel_c]))
        except Exception as e:
            logger.warning(f"Debug composite failed: {e}")
            cv2.imwrite(debug_path, debug_img)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_workpieces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect workpieces in the camera frame.

        Returns a list of dicts, each containing:
            id, class, confidence, angle_deg,
            box_px, corners_px, segmentation_px,
            corners_mm, segmentation_mm, box_mm, area_mm2
        """
        if image is None:
            return []

        from .calibration import calibration_service
        image = calibration_service.undistort(image)

        mean_brightness = float(np.mean(image))
        logger.debug(f"Input frame mean brightness: {mean_brightness:.1f}")
        if mean_brightness < 15:
            logger.warning(
                f"Frame very dark (mean={mean_brightness:.1f}). "
                "AE may still be settling — results may be unreliable."
            )

        results, raw_var_mask, proc_w, proc_h = self._detect_objects(image)
        self._save_debug_image(image, results, mean_brightness, raw_var_mask, proc_w, proc_h)
        return results


# Global instance
inference_service = InferenceService()
