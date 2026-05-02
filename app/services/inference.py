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
    """
    Simple, robust OpenCV workpiece detector for a honeycomb laser bed.

    The detection strategy is intentionally minimal:

      ① CLAHE-enhanced grayscale + Gaussian blur  (preprocessing)
      ② Thresholding to isolate the workpiece from the dark bed
            • Primary:  background subtraction diff vs empty-bed reference
            • Fallback: Otsu global threshold (no reference needed)
      ③ Convex-hull fill of each candidate blob
            Converts a partial / noisy blob into a solid convex shape,
            which naturally completes a workpiece seen at 60-80% coverage.
      ④ Morphological close to smooth hull edges
      ⑤ Contour area filter (min/max mm²)
      ⑥ minAreaRect → rotated bounding box + rotation angle
      ⑦ Homography map from pixels → physical mm coordinates

    Debug panels (calibration_data/detect_debug.jpg):
      A  RAW frame + final detection overlays
      B  CLAHE-enhanced input (what the algorithm sees)
      C  Threshold/diff mask — raw pixels before morphology (magenta)
      D  Candidate mask — after hull fill + close (yellow)  ← shape used for detection
      E  Canny edges — diagnostic only, not used for detection (cyan)
    """

    _REFERENCE_PATH = "calibration_data/reference_frame.jpg"

    def __init__(self):
        self.min_area_mm2       = float(os.getenv("MIN_WORKPIECE_AREA_MM2",   "500"))
        self.max_area_mm2       = float(os.getenv("MAX_WORKPIECE_AREA_MM2",   "160000"))
        self.honeycomb_kernel   = int(os.getenv("HONEYCOMB_KERNEL_SIZE",      "25"))
        self.apriltag_mask_pad  = int(os.getenv("APRILTAG_MASK_PADDING",      "30"))
        self.diff_threshold     = int(os.getenv("DETECT_DIFF_THRESHOLD",      "25"))
        # Cached reference frame (lazy-loaded at working resolution)
        self._ref_gray_proc: np.ndarray | None = None
        self._ref_proc_size: tuple[int, int] | None = None
        self.initialized = True
        logger.info("InferenceService initialised (threshold + hull-fill pipeline).")

    # ── Reference Frame ────────────────────────────────────────────────────────

    @property
    def has_reference(self) -> bool:
        return os.path.exists(self._REFERENCE_PATH)

    def save_reference_frame(self, image: np.ndarray) -> None:
        """Save an empty-bed image as the background subtraction reference."""
        os.makedirs(os.path.dirname(self._REFERENCE_PATH), exist_ok=True)
        cv2.imwrite(self._REFERENCE_PATH, image)
        self._ref_gray_proc = None
        self._ref_proc_size = None
        logger.info(f"Reference frame saved ({image.shape[1]}x{image.shape[0]}).")

    def clear_reference_frame(self) -> None:
        """Delete the reference frame (e.g. after lens recalibration)."""
        if os.path.exists(self._REFERENCE_PATH):
            os.remove(self._REFERENCE_PATH)
            logger.info("Reference frame deleted.")
        self._ref_gray_proc = None
        self._ref_proc_size = None

    def _load_ref_gray(self, proc_w: int, proc_h: int) -> np.ndarray | None:
        """Lazily load and CLAHE-enhance the reference frame at working resolution."""
        if not self.has_reference:
            return None
        if self._ref_gray_proc is not None and self._ref_proc_size == (proc_w, proc_h):
            return self._ref_gray_proc
        raw = cv2.imread(self._REFERENCE_PATH)
        if raw is None:
            logger.warning("Reference frame unreadable.")
            return None
        resized = cv2.resize(raw, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        # Store RAW grayscale — no CLAHE here.
        # CLAHE is content-adaptive: it normalises differently when a workpiece
        # covers the bed, making dark grain look 'unchanged' vs the empty reference.
        # Raw pixel values reliably reflect the physical brightness change.
        self._ref_gray_proc = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self._ref_proc_size = (proc_w, proc_h)
        logger.debug(f"Reference loaded at {proc_w}x{proc_h} (raw gray, no CLAHE).")
        return self._ref_gray_proc

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _odd(n: float) -> int:
        """Return the nearest odd integer ≥ 3 (OpenCV kernel requirement)."""
        n = max(3, int(n))
        return n if n % 2 == 1 else n + 1

    # ── Detection Core ─────────────────────────────────────────────────────────

    def _detect_objects(
        self, image: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Run the detection pipeline on an already-undistorted frame.

        Returns:
            (results, raw_thresh_mask, candidate_mask, canny_diag, proc_w, proc_h)
        """
        orig_h, orig_w = image.shape[:2]
        t0 = time.monotonic()

        # ── 1. Working resolution ─────────────────────────────────────────────
        max_dim  = int(os.getenv("DETECT_MAX_DIM", "1280"))
        long_side = max(orig_h, orig_w)
        if long_side > max_dim:
            ds     = max_dim / long_side
            proc_w = int(orig_w * ds)
            proc_h = int(orig_h * ds)
            proc   = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            ds     = 1.0
            proc_w, proc_h = orig_w, orig_h
            proc   = image

        logger.debug(f"Working res: {proc_w}×{proc_h}  ds={ds:.3f}")

        # ── 2. Preprocess ─────────────────────────────────────────────────────
        ref_dim  = 720.0
        scale    = min(proc_h, proc_w) / ref_dim
        smooth_k = self._odd(11 * scale)
        morph_k  = self._odd(self.honeycomb_kernel * scale)
        tag_pad  = int(self.apriltag_mask_pad * scale)
        min_area = max(100, int(1000 * scale * scale))

        gray     = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (smooth_k, smooth_k), 0)

        # ── 3. ROI mask from AprilTag convex hull ─────────────────────────────
        from .calibration import calibration_service
        tags_full = calibration_service.detect_tags(image, apply_undistort=False)
        tags_proc = [
            {**t, 'corners': (np.array(t['corners'], dtype=np.float32) * ds).tolist()}
            for t in tags_full
        ]
        logger.debug(f"AprilTags: {len(tags_proc)}")

        mask_roi = np.zeros((proc_h, proc_w), dtype=np.uint8)
        if len(tags_proc) >= 3:
            pts  = np.vstack([np.array(t['corners'], dtype=np.int32) for t in tags_proc])
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask_roi, [hull], 255)
        else:
            margin = int(80 * scale)
            mask_roi[margin:-margin, margin:-margin] = 255
            logger.warning(f"Only {len(tags_proc)} tags — using margin ROI.")

        # Erase tag squares (including white paper backing)
        for tag in tags_proc:
            corners  = np.array(tag['corners'], dtype=np.float32)
            centroid = corners.mean(axis=0)
            diag     = np.linalg.norm(corners - centroid, axis=1).max()
            scale_f  = 1.0 + tag_pad / max(diag, 1.0)
            expanded = centroid + (corners - centroid) * scale_f
            cv2.fillPoly(mask_roi, [expanded.astype(np.int32)], 0)

        # Erode ROI slightly to avoid boundary artefacts
        if tag_pad > 0:
            erode_se = cv2.getStructuringElement(
                cv2.MORPH_RECT, (tag_pad * 2 + 1, tag_pad * 2 + 1)
            )
            mask_roi = cv2.erode(mask_roi, erode_se, iterations=1)

        # ── 4. Threshold mask ─────────────────────────────────────────────────
        #
        #  PRIMARY — background subtraction diff:
        #    Works for ANY material (bright or dark) because it detects CHANGE,
        #    not absolute brightness.  Requires a reference frame captured with
        #    the bed empty (saved automatically during Mapping calibration).
        #
        #  FALLBACK — Otsu global threshold:
        #    The honeycomb bed is predominantly dark.  Most workpieces are
        #    lighter than the bed, so Otsu cleanly separates them.
        #    Works without a reference frame but misses dark materials.

        ref_gray = self._load_ref_gray(proc_w, proc_h)

        if ref_gray is not None:
            # Diff on RAW blurred grayscale — NOT the CLAHE-enhanced image.
            # ref_gray is raw (no CLAHE); we blur both sides equally to suppress
            # per-pixel camera noise before thresholding the change magnitude.
            gray_blurred = cv2.GaussianBlur(gray, (smooth_k, smooth_k), 0)
            ref_blurred  = cv2.GaussianBlur(ref_gray, (smooth_k, smooth_k), 0)
            abs_diff     = cv2.absdiff(gray_blurred, ref_blurred)
            _, raw_mask  = cv2.threshold(
                abs_diff, self.diff_threshold, 255, cv2.THRESH_BINARY
            )
            strategy = "diff"
            logger.debug(
                f"Diff mask (raw gray): threshold={self.diff_threshold}  "
                f"px={cv2.countNonZero(raw_mask)}"
            )
        else:
            # Otsu on the ROI pixels only
            roi_pixels = blurred[mask_roi > 0]
            if len(roi_pixels) > 0:
                otsu_t, _ = cv2.threshold(
                    roi_pixels, 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )
            else:
                otsu_t = 127
            _, raw_mask = cv2.threshold(blurred, otsu_t, 255, cv2.THRESH_BINARY)
            strategy = f"otsu@{otsu_t:.0f}"
            logger.debug(f"Otsu mask: threshold={otsu_t:.0f}")

        # Restrict to ROI
        raw_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=mask_roi)

        # ── 5. Combined convex hull of all valid fragments ────────────────────
        #
        #  Wood grain / uneven lighting splits the threshold mask into several
        #  disconnected blobs.  If we hull each blob individually the result
        #  is several small hulls that don't cover the whole piece.
        #
        #  Solution: combine all valid fragment points into ONE point cloud and
        #  take a single convex hull.  Even if grain cuts the piece into three
        #  fragments the combined hull spans from the top-left fragment to the
        #  bottom-right fragment — giving the correct overall extent.
        #
        #  Fragments below min_area * 0.1 are still accepted as contributors
        #  (they help define the outer boundary) but below min_area * 0.05 are
        #  pure noise and ignored.

        init_cnts, _ = cv2.findContours(
            raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        fragment_pts = []
        for cnt in init_cnts:
            if cv2.contourArea(cnt) >= max(min_area * 0.05, 30):
                fragment_pts.append(cnt.reshape(-1, 2))

        hull_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
        if fragment_pts:
            combined_pts  = np.vstack(fragment_pts)
            combined_hull = cv2.convexHull(combined_pts)
            cv2.fillPoly(hull_mask, [combined_hull], 255)

        # ── 6. Edge-pad dilation + morphological close ────────────────────────
        #
        #  The combined hull may still fall short of the real workpiece boundary
        #  if the threshold missed a dark-grained ring around the edges.
        #  DETECT_EDGE_PAD_PX (default = morph_k) expands the hull outward to
        #  cover those missed edge pixels before the final close.
        edge_pad = int(os.getenv("DETECT_EDGE_PAD_PX", "0"))
        if edge_pad > 0:
            pad_se   = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self._odd(edge_pad), self._odd(edge_pad))
            )
            hull_mask = cv2.dilate(hull_mask, pad_se, iterations=1)

        close_k  = self._odd(morph_k * 2)
        close_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        candidate_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, close_se)

        # Restrict to ROI
        candidate_mask = cv2.bitwise_and(candidate_mask, candidate_mask, mask=mask_roi)

        # ── 7. Canny — diagnostic only ────────────────────────────────────────
        canny_h = float(os.getenv("DETECT_CANNY_HIGH", "60"))
        canny_diag = cv2.Canny(blurred, canny_h * 0.2, canny_h)
        canny_diag = cv2.bitwise_and(canny_diag, canny_diag, mask=mask_roi)

        # ── 8. Final contours from candidate mask ─────────────────────────────
        final_cnts, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.debug(f"Candidate contours: {len(final_cnts)}  strategy={strategy}")

        # ── 9. Build results ───────────────────────────────────────────────────
        results: List[Dict] = []
        from .calibration import calibration_service as cal

        for i, cnt in enumerate(final_cnts):
            # Pixel-space area filter (cheap, pre-mm)
            if cv2.contourArea(cnt) < min_area:
                continue
            if cv2.contourArea(cnt) > proc_w * proc_h * 0.5:
                continue

            # Scale back to full-res pixel coordinates
            cnt_full = (cnt.astype(np.float64) / ds).astype(np.int32) if ds < 1.0 else cnt

            # Rotated bounding box
            rect       = cv2.minAreaRect(cnt_full)
            corners_px = cv2.boxPoints(rect).astype(np.float32)

            # Rotation angle: pick the longest edge, compute with atan2.
            # This is version-agnostic (minAreaRect angle conventions differ
            # between OpenCV versions) and normalises to [-45, 45].
            edges    = [corners_px[1] - corners_px[0], corners_px[2] - corners_px[1]]
            long_e   = max(edges, key=lambda e: float(e[0]**2 + e[1]**2))
            angle_deg = math.degrees(math.atan2(float(long_e[1]), float(long_e[0])))
            while angle_deg >  45.0: angle_deg -= 90.0
            while angle_deg < -45.0: angle_deg += 90.0

            corners_px = corners_px.astype(np.int32)

            # Convex hull of full-res contour for segmentation polygon
            hull_full = cv2.convexHull(cnt_full)
            seg_px    = hull_full.reshape(-1, 2).tolist()
            x, y, w, h = cv2.boundingRect(cnt_full)

            # Map to physical mm via homography
            corners_mm = cal.map_pixels_to_mm(corners_px.astype(np.float32))
            seg_mm     = cal.map_pixels_to_mm(np.array(seg_px, dtype=np.float32))

            area_mm2 = float(
                cv2.contourArea(corners_mm.reshape(-1, 1, 2).astype(np.float32))
            )
            if not (self.min_area_mm2 <= area_mm2 <= self.max_area_mm2):
                logger.debug(
                    f"Filtered {i}: area={area_mm2:.0f} mm²  "
                    f"range=[{self.min_area_mm2}, {self.max_area_mm2}]"
                )
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

        elapsed = time.monotonic() - t0
        logger.info(f"Detection: {len(results)} workpiece(s) in {elapsed:.3f}s")
        return results, raw_mask, candidate_mask, canny_diag, proc_w, proc_h

    # ── Debug Composite ────────────────────────────────────────────────────────

    def _save_debug_image(
        self,
        image: np.ndarray,
        results: List[Dict],
        mean_brightness: float,
        raw_mask: np.ndarray | None,
        candidate_mask: np.ndarray | None,
        canny_diag: np.ndarray | None,
        proc_w: int,
        proc_h: int,
    ) -> None:
        """
        Write a 5-panel 3+2 grid to calibration_data/detect_debug.jpg:

          [A] Raw frame + overlays  [B] CLAHE input  [C] Threshold/diff mask
          [D] Candidate mask (after hull fill + close)  [E] Canny (diag only)
        """
        debug_path = "calibration_data/detect_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        overlay = image.copy()
        for wp in results:
            pts     = np.array(wp['corners_px'], dtype=np.int32)
            cx      = int(pts[:, 0].mean())
            cy      = int(pts[:, 1].mean())
            angle_r = math.radians(wp.get('angle_deg', 0))
            ax      = int(cx + 50 * math.cos(angle_r))
            ay      = int(cy + 50 * math.sin(angle_r))

            cv2.polylines(overlay, [pts], True, (0, 220, 0), 2)
            cv2.arrowedLine(overlay, (cx, cy), (ax, ay), (0, 140, 255), 2, tipLength=0.25)

            def _text(img, txt, pos, scale, fg, thickness):
                """Draw outlined text: thick black stroke then bright colour on top."""
                cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, fg,       thickness,     cv2.LINE_AA)

            _text(overlay,
                  f"{wp['id']}  {wp.get('angle_deg', 0):.1f}\u00b0  {wp.get('area_mm2', 0):.0f}mm\u00b2",
                  (pts[0][0], pts[0][1] - 12),
                  0.55, (0, 255, 60), 2)

            for px, mm in zip(pts, wp['corners_mm']):
                _text(overlay,
                      f"({int(mm[0])},{int(mm[1])})",
                      (int(px[0]), int(px[1])),
                      0.6, (255, 255, 255), 2)

        try:
            MAX_W = 1280
            h, w  = overlay.shape[:2]
            s     = min(1.0, MAX_W / w)
            tw, th = int(w * s), int(h * s)

            panel_a = cv2.resize(overlay, (tw, th))

            # CLAHE base — reused as the background for panels B, C, D, E
            gray_b   = cv2.cvtColor(cv2.resize(image, (tw, th)), cv2.COLOR_BGR2GRAY)
            clahe_b  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray_b)
            base_bgr = cv2.cvtColor(clahe_b, cv2.COLOR_GRAY2BGR)
            panel_b  = base_bgr.copy()

            def _overlay(mask: np.ndarray | None, color_bgr: tuple,
                         alpha: float = 0.55) -> np.ndarray:
                """
                Blend a coloured mask on top of the CLAHE base image.
                Masked pixels = (1-alpha)*base + alpha*colour.
                Unmasked pixels = base (the scene is fully visible).
                """
                out = base_bgr.copy()
                if mask is None or mask.size == 0:
                    return out
                m = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                color_layer = np.full((th, tw, 3), color_bgr, dtype=np.uint8)
                where = m > 128
                out[where] = cv2.addWeighted(
                    base_bgr, 1 - alpha, color_layer, alpha, 0
                )[where]
                return out

            panel_c = _overlay(raw_mask,      (255,  50, 180))   # magenta — raw threshold
            panel_d = _overlay(candidate_mask, (0,   230, 230))   # yellow  — candidates
            panel_e = _overlay(canny_diag,     (0,   255, 100))   # green   — canny diag

            lbl = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.5, color=(0, 255, 255), thickness=2)
            has_ref = "ref" if self.has_reference else "no-ref"
            cv2.putText(panel_a, f"A: RAW  mean={mean_brightness:.0f}", (8, 24), **lbl)
            cv2.putText(panel_b, "B: CLAHE input", (8, 24), **lbl)
            cv2.putText(panel_c, f"C: Threshold ({has_ref})", (8, 24), **lbl)
            cv2.putText(panel_d, "D: Candidates (hull+close)", (8, 24), **lbl)
            cv2.putText(panel_e, "E: Canny (diag only)", (8, 24), **lbl)

            top_row    = np.hstack([panel_a, panel_b, panel_c])
            pad_w      = top_row.shape[1] - tw * 2
            bottom_row = np.hstack([panel_d, panel_e,
                                    np.zeros((th, pad_w, 3), dtype=np.uint8)])
            cv2.imwrite(debug_path, np.vstack([top_row, bottom_row]))
        except Exception as exc:
            logger.warning(f"Debug composite failed: {exc}")
            cv2.imwrite(debug_path, overlay)

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_workpieces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect workpieces in a camera frame.

        Returns a list of dicts containing:
            id, class, confidence, angle_deg,
            box_px, corners_px, segmentation_px,
            corners_mm, segmentation_mm, box_mm, area_mm2
        """
        if image is None:
            return []

        from .calibration import calibration_service
        image = calibration_service.undistort(image)

        mean_brightness = float(np.mean(image))
        if mean_brightness < 15:
            logger.warning(
                f"Frame very dark (mean={mean_brightness:.1f}) — "
                "AE may still be settling."
            )

        results, raw_mask, candidate_mask, canny_diag, pw, ph = \
            self._detect_objects(image)

        self._save_debug_image(
            image, results, mean_brightness,
            raw_mask, candidate_mask, canny_diag, pw, ph,
        )
        return results


# Singleton
inference_service = InferenceService()
