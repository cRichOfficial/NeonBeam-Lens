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

    Detection strategy: **Background subtraction + local texture variance (hybrid)**

    Primary:  Absolute diff against a stored reference frame of the empty bed.
              When a workpiece is placed, it changes the pixels it covers.
              The diff cleanly isolates any placed object regardless of material.

    Secondary: Local texture variance within the ROI catches objects that were
               present when the reference was captured or in the absence of a
               reference frame.

    The two masks are OR-combined so detection is robust even without a reference.
    All expensive processing runs on a downsampled working image (capped at
    DETECT_MAX_DIM px on the long side).  Contour coordinates are scaled back
    to full resolution before the calibration homography maps them to physical
    mm values.
    """

    _REFERENCE_PATH = "calibration_data/reference_frame.jpg"

    def __init__(self):
        self.min_area_mm2 = float(os.getenv("MIN_WORKPIECE_AREA_MM2", "500"))
        self.max_area_mm2 = float(os.getenv("MAX_WORKPIECE_AREA_MM2", "160000"))
        self.honeycomb_kernel = int(os.getenv("HONEYCOMB_KERNEL_SIZE", "25"))
        self.apriltag_mask_padding = int(os.getenv("APRILTAG_MASK_PADDING", "30"))
        self.variance_window = int(os.getenv("DETECT_VARIANCE_WINDOW", "55"))
        self.variance_percentile = float(os.getenv("DETECT_VARIANCE_PERCENTILE", "70"))
        self.variance_multiplier = float(os.getenv("DETECT_VARIANCE_MULTIPLIER", "0.35"))
        # Diff threshold: pixel change (0-255) that counts as "something moved here"
        self.diff_threshold = int(os.getenv("DETECT_DIFF_THRESHOLD", "25"))
        # Cached reference frame (proc-res CLAHE grayscale, loaded lazily)
        self._ref_gray_proc: np.ndarray | None = None
        self._ref_proc_size: tuple[int, int] | None = None  # (w, h)
        self.initialized = True
        logger.info("InferenceService initialised (background-subtraction + variance hybrid).")

    # ── Reference Frame ────────────────────────────────────────────────────────

    @property
    def has_reference(self) -> bool:
        return os.path.exists(self._REFERENCE_PATH)

    def save_reference_frame(self, image: np.ndarray) -> None:
        """
        Save `image` (already undistorted full-res frame) as the empty-bed reference.
        The image is stored at full resolution so it can be re-processed at any
        working resolution in future detection runs.
        """
        os.makedirs(os.path.dirname(self._REFERENCE_PATH), exist_ok=True)
        cv2.imwrite(self._REFERENCE_PATH, image)
        # Invalidate cached proc-res copy
        self._ref_gray_proc = None
        self._ref_proc_size = None
        logger.info(f"Reference frame saved ({image.shape[1]}x{image.shape[0]}).")

    def clear_reference_frame(self) -> None:
        """Delete the saved reference frame (e.g. after lens recalibration)."""
        if os.path.exists(self._REFERENCE_PATH):
            os.remove(self._REFERENCE_PATH)
            logger.info("Reference frame deleted.")
        self._ref_gray_proc = None
        self._ref_proc_size = None

    def _load_ref_gray(self, proc_w: int, proc_h: int) -> np.ndarray | None:
        """
        Return the reference frame as a CLAHE-enhanced grayscale image at
        (proc_w, proc_h) resolution. Result is cached until the file changes.
        """
        if not self.has_reference:
            return None
        if self._ref_gray_proc is not None and self._ref_proc_size == (proc_w, proc_h):
            return self._ref_gray_proc
        ref_bgr = cv2.imread(self._REFERENCE_PATH)
        if ref_bgr is None:
            logger.warning("Reference frame file missing or unreadable.")
            return None
        ref_resized = cv2.resize(ref_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._ref_gray_proc = clahe.apply(ref_gray)
        self._ref_proc_size = (proc_w, proc_h)
        logger.debug(f"Reference frame loaded at {proc_w}x{proc_h}.")
        return self._ref_gray_proc


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

        # Gaussian blur is used for morphological pre-processing (smooth_k) but
        # we compute local variance on the RAW enhanced image so that honeycomb
        # cell-wall transitions are preserved. Blurring before variance washes
        # out those transitions and makes cell interiors indistinguishable from
        # solid workpiece surfaces.
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

        # ── 4a. Background subtraction (primary signal) ──────────────────────
        #   If a reference frame of the empty bed has been saved, compute the
        #   absolute per-pixel diff between the current CLAHE image and the
        #   reference CLAHE image.  Pixels that changed significantly are where
        #   a workpiece has been placed.
        ref_gray = self._load_ref_gray(proc_w, proc_h)
        if ref_gray is not None:
            abs_diff = cv2.absdiff(enhanced, ref_gray)
            # Blur before thresholding to reduce isolated noise pixels
            abs_diff_blurred = cv2.GaussianBlur(abs_diff, (smooth_k, smooth_k), 0)
            _, diff_mask_raw = cv2.threshold(
                abs_diff_blurred, self.diff_threshold, 255, cv2.THRESH_BINARY
            )
            diff_mask = cv2.bitwise_and(diff_mask_raw, diff_mask_raw, mask=mask_roi)
            logger.debug(
                f"Background diff: threshold={self.diff_threshold}, "
                f"changed_px={cv2.countNonZero(diff_mask)}"
            )
        else:
            diff_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
            logger.debug("No reference frame — skipping background subtraction.")

        # ── 4b. Local variance on the UNSMOOTHED CLAHE image ─────────────────
        #    Variance = E[X²] − E[X]²   (two box filters, very fast)
        #
        #    var_k must span 2+ honeycomb cells so the window sees wall
        #    transitions and marks cell regions as HIGH variance.
        #    Honeycomb → high variance (cell walls create sharp transitions)
        #    Solid workpiece → low variance (continuous uniform surface)
        #
        #    Using `enhanced` (not `smoothed`) preserves the cell-wall signal.
        sf = enhanced.astype(np.float32)
        mean_img = cv2.blur(sf, (var_k, var_k))
        mean_sq  = cv2.blur(sf * sf, (var_k, var_k))
        local_var = np.maximum(mean_sq - mean_img * mean_img, 0.0)

        # ── 5. Threshold: low-variance regions inside ROI ────────────────────
        roi_var = local_var[mask_roi > 0]
        if len(roi_var) == 0:
            logger.warning("ROI is empty — cannot detect objects.")
            empty = np.zeros((proc_h, proc_w), dtype=np.uint8)
            return [], empty, empty, empty, proc_w, proc_h

        hc_baseline = float(np.percentile(roi_var, self.variance_percentile))
        var_thresh   = hc_baseline * self.variance_multiplier

        logger.debug(
            f"Variance: baseline(p{self.variance_percentile:.0f})={hc_baseline:.1f}, "
            f"threshold={var_thresh:.1f}"
        )

        low_var = (local_var < var_thresh).astype(np.uint8) * 255
        var_mask_roi = cv2.bitwise_and(low_var, low_var, mask=mask_roi)
        raw_var_mask = var_mask_roi.copy()  # for debug panel

        # ── 5b. Combine diff + variance (OR) ────────────────────────────────
        #   OR gives maximum recall: detect anything that changed OR has
        #   anomalous texture (useful when reference isn’t perfectly matched).
        if ref_gray is not None:
            workpiece_mask = cv2.bitwise_or(diff_mask, var_mask_roi)
            logger.debug("Using diff OR variance combined mask.")
        else:
            workpiece_mask = var_mask_roi
            logger.debug("Using variance-only mask (no reference).")


        # ── 6. Two-pass morphology on variance mask ──────────────────────────
        #   Pass A — small close: fills individual cell holes within a blob
        morph_se = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        closed = cv2.morphologyEx(workpiece_mask, cv2.MORPH_CLOSE, morph_se)

        #   Pass B — large close: merges fragments into a single workpiece blob
        large_k = self._odd(morph_k * 3)
        large_se = cv2.getStructuringElement(cv2.MORPH_RECT, (large_k, large_k))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, large_se)

        #   Erode boundary artefacts, then dilate back to true size
        boundary_se = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        eroded  = cv2.erode(closed, boundary_se, iterations=1)
        dilated = cv2.dilate(eroded, boundary_se, iterations=2)

        # ── 7. Candidate blobs from variance mask ────────────────────────────
        var_contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.debug(f"Variance blobs before area filter: {len(var_contours)}")

        # ── 8. Edge-refinement stage ─────────────────────────────────────────
        #
        #   Strategy: run Canny on the lightly-smoothed CLAHE image, then
        #   restrict it to a "boundary ring" built from the variance blob.
        #
        #   The ring is: dilate(blob) − blob  ← just the perimeter pixels.
        #   Because the ring only covers the transition zone between the
        #   workpiece and the bed, honeycomb edges (which are far outside
        #   the blob) are never even considered by Canny.
        #
        #   This avoids the blur-vs-cell-edges arms race entirely.

        # Light blur — just enough to reduce sensor noise for Canny
        canny_blur_k = self._odd(smooth_k)
        canny_blurred = cv2.GaussianBlur(enhanced, (canny_blur_k, canny_blur_k), 0)

        canny_high = float(os.getenv("DETECT_CANNY_HIGH", "60"))
        canny_low  = canny_high * 0.2
        canny_full = cv2.Canny(canny_blurred, canny_low, canny_high)
        canny_full = cv2.bitwise_and(canny_full, canny_full, mask=mask_roi)

        # Build boundary ring around the morphologically cleaned mask
        ring_k  = self._odd(large_k)   # ring width ~ one workpiece-merge kernel
        ring_se = cv2.getStructuringElement(cv2.MORPH_RECT, (ring_k, ring_k))
        blob_dilated = cv2.dilate(dilated, ring_se, iterations=1)
        blob_ring    = cv2.subtract(blob_dilated, dilated)  # perimeter only

        # Restrict Canny to the boundary ring — these are the workpiece edges
        boundary_canny = cv2.bitwise_and(canny_full, canny_full, mask=blob_ring)

        # Dilate boundary edges slightly to close any gaps in the outline
        gap_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        boundary_canny_closed = cv2.dilate(boundary_canny, gap_se, iterations=1)

        # Store for debug panel D (show full Canny + ring overlay)
        debug_canny = canny_full.copy()
        # Tint the active ring brighter so it's visible alongside the full map
        debug_canny = np.where(blob_ring[..., None] > 0,
                               np.full_like(debug_canny[..., None], 200),
                               debug_canny[..., None]).squeeze()
        debug_canny = debug_canny.astype(np.uint8)

        results = []
        from .calibration import calibration_service as cal

        for i, var_cnt in enumerate(var_contours):
            area_proc = cv2.contourArea(var_cnt)
            if area_proc < min_area or area_proc > (proc_w * proc_h * 0.5):
                continue

            # Create a single-blob mask for this contour
            single_blob = np.zeros((proc_h, proc_w), dtype=np.uint8)
            cv2.fillPoly(single_blob, [var_cnt], 255)

            # Build boundary ring for this specific blob
            blob_ring_single = cv2.dilate(single_blob, ring_se, iterations=1)
            blob_ring_single = cv2.subtract(blob_ring_single, single_blob)

            # Canny edges restricted to this blob's boundary ring
            region_edges = cv2.bitwise_and(
                boundary_canny_closed, boundary_canny_closed, mask=blob_ring_single
            )

            edge_cnts, _ = cv2.findContours(
                region_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Pick the largest edge contour that is meaningful
            best_edge_cnt = None
            if edge_cnts:
                edge_cnts_sorted = sorted(edge_cnts, key=cv2.contourArea, reverse=True)
                candidate = edge_cnts_sorted[0]
                if cv2.contourArea(candidate) >= min_area * 0.3:
                    best_edge_cnt = candidate

            if best_edge_cnt is not None:
                # Combine the variance blob fill + the refined boundary contour
                # so the final shape covers the full interior, not just the ring
                refined_mask = single_blob.copy()
                cv2.drawContours(refined_mask, [best_edge_cnt], -1, 255,
                                 thickness=cv2.FILLED)
                refined_cnts, _ = cv2.findContours(
                    refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                refined_cnt = max(refined_cnts, key=cv2.contourArea) \
                    if refined_cnts else var_cnt
                logger.debug(
                    f"Blob {i}: edge-refined "
                    f"({cv2.contourArea(best_edge_cnt):.0f} px² ring)"
                )
            else:
                refined_cnt = var_cnt
                logger.debug(f"Blob {i}: using variance blob (no boundary edges found)")

            # ── Scale to full-res and map to mm ─────────────────────────────
            if ds < 1.0:
                cnt_full = (refined_cnt.astype(np.float64) / ds).astype(np.int32)
            else:
                cnt_full = refined_cnt

            rect = cv2.minAreaRect(cnt_full)
            corners_px = cv2.boxPoints(rect).astype(np.int32)
            angle_deg  = float(rect[2])

            hull = cv2.convexHull(cnt_full)
            seg_px = hull.reshape(-1, 2).tolist()

            x, y, w, h = cv2.boundingRect(cnt_full)

            corners_mm = cal.map_pixels_to_mm(corners_px.astype(np.float32))
            seg_mm     = cal.map_pixels_to_mm(np.array(seg_px, dtype=np.float32))

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
        return results, raw_var_mask, diff_mask, debug_canny, proc_w, proc_h

    # ── Debug Image ────────────────────────────────────────────────────────────

    def _save_debug_image(
        self, image: np.ndarray, results: List[Dict],
        mean_brightness: float,
        var_mask_proc: np.ndarray | None = None,
        diff_mask_proc: np.ndarray | None = None,
        canny_proc: np.ndarray | None = None,
        proc_w: int = 0, proc_h: int = 0,
    ) -> None:
        """
        Save a 5-panel diagnostic composite (3-top + 2-bottom grid):
          [A] RAW frame + overlays   [B] CLAHE input   [C] Variance mask (cyan)
          [D] Background diff (magenta)                [E] Canny edges (yellow)
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

            # Panel C: variance mask
            if var_mask_proc is not None and var_mask_proc.size > 0:
                panel_c = cv2.cvtColor(
                    cv2.resize(var_mask_proc, (tw, th), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR
                )
                cyan_tint = np.zeros_like(panel_c)
                cyan_tint[panel_c[:, :, 0] > 128] = [255, 255, 0]
                panel_c = cv2.addWeighted(panel_c, 0.4, cyan_tint, 0.6, 0)
            else:
                panel_c = np.zeros((th, tw, 3), dtype=np.uint8)

            # Panel D: background diff (magenta = changed pixels)
            if diff_mask_proc is not None and diff_mask_proc.size > 0:
                panel_d = cv2.cvtColor(
                    cv2.resize(diff_mask_proc, (tw, th), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR
                )
                mag_tint = np.zeros_like(panel_d)
                mag_tint[panel_d[:, :, 0] > 128] = [255, 0, 255]  # magenta
                panel_d = cv2.addWeighted(panel_d, 0.0, mag_tint, 1.0, 0)
            else:
                panel_d = np.zeros((th, tw, 3), dtype=np.uint8)
                cv2.putText(panel_d, "No reference frame", (10, th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

            # Panel E: Canny boundary edges (yellow)
            if canny_proc is not None and canny_proc.size > 0:
                panel_e = cv2.cvtColor(
                    cv2.resize(canny_proc, (tw, th), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR
                )
                yellow_tint = np.zeros_like(panel_e)
                yellow_tint[panel_e[:, :, 0] > 128] = [0, 255, 255]
                panel_e = cv2.addWeighted(panel_e, 0.0, yellow_tint, 1.0, 0)
            else:
                panel_e = np.zeros((th, tw, 3), dtype=np.uint8)

            lbl = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55,
                       color=(0, 255, 255), thickness=2)
            cv2.putText(panel_a, f"A: RAW  mean={mean_brightness:.0f}", (10, 28), **lbl)
            cv2.putText(panel_b, "B: CLAHE input", (10, 28), **lbl)
            cv2.putText(panel_c, "C: Variance (cyan)", (10, 28), **lbl)
            has_ref = "YES" if (diff_mask_proc is not None and diff_mask_proc.any()) else "NO REF"
            cv2.putText(panel_d, f"D: Diff/BG sub ({has_ref})", (10, 28), **lbl)
            cv2.putText(panel_e, "E: Canny boundary", (10, 28), **lbl)

            # 3-top + 2-bottom layout
            top_row    = np.hstack([panel_a, panel_b, panel_c])
            # Pad D and E to match top_row width
            pad_w = top_row.shape[1] - tw * 2
            pad = np.zeros((th, pad_w, 3), dtype=np.uint8)
            bottom_row = np.hstack([panel_d, panel_e, pad])
            cv2.imwrite(debug_path, np.vstack([top_row, bottom_row]))
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

        results, raw_var_mask, diff_mask, debug_canny, proc_w, proc_h = self._detect_objects(image)
        self._save_debug_image(
            image, results, mean_brightness,
            raw_var_mask, diff_mask, debug_canny, proc_w, proc_h
        )
        return results


# Global instance
inference_service = InferenceService()
