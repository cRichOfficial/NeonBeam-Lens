import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from .calibration import calibration_service

logger = logging.getLogger("vision_api.inference")

# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8-seg post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Simple NMS — returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return kept


def _decode_yolov8_seg(outputs: Dict[str, np.ndarray], orig_h: int, orig_w: int,
                        conf_threshold: float = 0.4, iou_threshold: float = 0.45,
                        model_size: int = 640) -> List[Dict]:
    """
    Decode raw un-fused YOLOv8-seg tensors from the Hailo hardware.
    The hardware returns 10 tensors: 3 scales (80x80, 40x40, 20x20) * 3 branches 
    (box DFL, class scores, mask coeffs) + 1 prototype mask tensor.
    """
    scales = {80: {}, 40: {}, 20: {}}
    proto_tensor = None

    # Sort tensors by spatial grid size and feature type
    for name, tensor in outputs.items():
        t = tensor.squeeze(0)  # Remove batch dim: (H, W, C)
        h, w, c = t.shape
        if h == 160 and w == 160 and c == 32:
            proto_tensor = t.transpose(2, 0, 1)  # Convert to (32, 160, 160) for matmul
            continue
            
        if h in scales:
            if c == 80:
                # Sigmoid is usually applied in hardware for class scores, but let's be safe
                # If values are > 1, it needs sigmoid. Usually hailort normalizes it.
                scales[h]['cls'] = t 
            elif c == 64:
                scales[h]['reg'] = t
            elif c == 32:
                scales[h]['mask'] = t

    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_mask_coeffs = []

    dfl_weights = np.arange(16, dtype=np.float32)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    for grid_size, tensors in scales.items():
        if 'cls' not in tensors or 'reg' not in tensors:
            continue
            
        stride = model_size / grid_size
        cls_t = tensors['cls']
        reg_t = tensors['reg']
        mask_t = tensors.get('mask')

        # Find anchors exceeding confidence threshold
        max_scores = np.max(cls_t, axis=-1)
        # Apply sigmoid if hardware didn't (Hailo usually outputs logits if un-fused)
        if np.max(max_scores) > 1.0 or np.min(max_scores) < 0.0:
            max_scores = _sigmoid(max_scores)
            cls_t = _sigmoid(cls_t)

        logger.info(f"Scale {grid_size}x{grid_size}: Absolute highest confidence score is {float(np.max(max_scores)):.4f}")

        valid_mask = max_scores > conf_threshold
        if not np.any(valid_mask):
            continue

        y_idx, x_idx = np.where(valid_mask)
        scores = max_scores[valid_mask]
        class_ids = np.argmax(cls_t[valid_mask], axis=-1)
        
        reg_features = reg_t[valid_mask].reshape(-1, 4, 16)
        mask_coeffs = mask_t[valid_mask] if mask_t is not None else np.zeros((len(scores), 32))

        # Decode DFL (Distribution Focal Loss)
        dist = softmax(reg_features)
        pred_dist = np.sum(dist * dfl_weights, axis=-1)

        anchor_x = (x_idx + 0.5) * stride
        anchor_y = (y_idx + 0.5) * stride

        x1 = anchor_x - pred_dist[:, 0] * stride
        y1 = anchor_y - pred_dist[:, 1] * stride
        x2 = anchor_x + pred_dist[:, 2] * stride
        y2 = anchor_y + pred_dist[:, 3] * stride

        boxes = np.stack([x1, y1, x2, y2], axis=-1)

        logger.info(f"Scale {grid_size}x{grid_size}: {len(scores)} anchors passed confidence > {conf_threshold}")

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)
        all_mask_coeffs.append(mask_coeffs)

    if not all_boxes:
        logger.warning("No anchors passed confidence threshold across all scales.")
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_class_ids, axis=0)
    mask_coeffs = np.concatenate(all_mask_coeffs, axis=0)
    
    logger.info(f"Total raw detections before NMS: {len(boxes)}")

    # Scale to original image
    scale_x = orig_w / model_size
    scale_y = orig_h / model_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    boxes = np.clip(boxes, 0, [[orig_w, orig_h, orig_w, orig_h]])

    # NMS
    kept = _nms(boxes, scores, iou_threshold)
    logger.info(f"Total detections after NMS: {len(kept)}")
    
    results = []

    for idx in kept:
        x1, y1, x2, y2 = boxes[idx].astype(int)

        if proto_tensor is not None:
            # Mask generation: dot product of coefficients with prototype masks
            ph, pw = proto_tensor.shape[1], proto_tensor.shape[2]
            mask_logits = (mask_coeffs[idx] @ proto_tensor.reshape(32, -1)).reshape(ph, pw)
            instance_mask = (_sigmoid(mask_logits) > 0.5).astype(np.uint8)
            instance_mask = cv2.resize(instance_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            instance_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            instance_mask[y1:y2, x1:x2] = 1

        results.append({
            "_box": [x1, y1, x2, y2],
            "_mask": instance_mask,
            "_confidence": float(scores[idx]),
            "_class_id": int(class_ids[idx]),
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Inference Service
# ──────────────────────────────────────────────────────────────────────────────

class InferenceService:
    def __init__(self):
        self.use_hailo = os.getenv("USE_HAILO", "False").lower() == "true"
        self.model_path = os.getenv("MODEL_PATH", "app/models/yolov8s_seg.hef")
        self.conf_threshold = float(os.getenv("DETECT_CONF_THRESHOLD", "0.4"))
        self.min_area_mm2 = float(os.getenv("MIN_WORKPIECE_AREA_MM2", "500"))
        self.max_area_mm2 = float(os.getenv("MAX_WORKPIECE_AREA_MM2", "160000"))
        self.honeycomb_kernel = int(os.getenv("HONEYCOMB_KERNEL_SIZE", "25"))
        self.model_input_size = 640

        self.vdevice = None
        self.network_group = None
        self.input_vstream_infos = None
        self.output_vstream_infos = None
        self.initialized = False

        if self.use_hailo:
            self._init_hailo()
        else:
            logger.info("USE_HAILO=False — using OpenCV software detection fallback.")
            self.initialized = True

    # ── Hailo Initialization ──────────────────────────────────────────────────

    def _init_hailo(self):
        """Initialize Hailo-8L NPU via hailort Python API."""
        import sys
        if "/usr/lib/python3/dist-packages" not in sys.path:
            sys.path.append("/usr/lib/python3/dist-packages")

        try:
            try:
                import hailort
            except ImportError:
                # The Debian apt package exposes VDevice at the root of hailo_platform
                import hailo_platform as hailort

            logger.info(f"Loading Hailo HEF from {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"HEF not found at '{self.model_path}'. "
                    f"Run setup_native.sh to install the model. See app/models/README.md."
                )

            self.vdevice = hailort.VDevice()
            hef = hailort.HEF(self.model_path)
            configured = self.vdevice.configure(hef)
            self.network_group = configured[0]
            self.input_vstream_infos = hef.get_input_vstream_infos()
            self.output_vstream_infos = hef.get_output_vstream_infos()
            self.initialized = True
            logger.info("Hailo NPU initialized successfully.")

        except ImportError as e:
            raise ImportError(
                f"hailort Python package not found ({e}). "
                "Install HailoRT Python bindings (.whl) from the Hailo Developer Zone."
            )
        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Hailo NPU: {e}")

    # ── Hailo Inference ───────────────────────────────────────────────────────

    def _run_hailo_inference(self, image: np.ndarray) -> List[Dict]:
        """Run YOLOv8-seg on the Hailo NPU and return raw pre-mapped detections."""
        import sys
        if "/usr/lib/python3/dist-packages" not in sys.path:
            sys.path.append("/usr/lib/python3/dist-packages")
        
        try:
            import hailort
        except ImportError:
            import hailo_platform as hailort

        orig_h, orig_w = image.shape[:2]
        size = self.model_input_size

        # Preprocess: resize to 640×640, ensure uint8 RGB
        input_img = cv2.resize(image, (size, size))
        if len(input_img.shape) == 2:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
        elif input_img.shape[2] == 4:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2RGB)
        else:
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.uint8)

        input_params = hailort.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hailort.FormatType.UINT8
        )
        output_params = hailort.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hailort.FormatType.FLOAT32
        )

        network_group_params = self.network_group.create_params()
        with self.network_group.activate(network_group_params):
            with hailort.InferVStreams(self.network_group, input_params, output_params) as pipeline:
                input_data = {self.input_vstream_infos[0].name: np.expand_dims(input_img, 0)}
                raw_outputs = pipeline.infer(input_data)

        detections = _decode_yolov8_seg(
            raw_outputs, orig_h, orig_w,
            conf_threshold=self.conf_threshold,
            model_size=size,
        )
        return self._build_workpiece_list(detections, orig_h, orig_w)

    def _software_detect(self, image: np.ndarray) -> List[Dict]:
        """
        OpenCV-based fallback detector optimized for a black honeycomb laser bed.
        Uses CLAHE-enhanced Red channel and heavy morphological closing.
        """
        orig_h, orig_w = image.shape[:2]
        
        # 1. Use the Red channel
        red = image[:, :, 2]

        # 2. Enhance contrast to find dark slate on dark bed
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(red)

        # 3. Aggressive Blur to merge honeycomb texture
        blurred = cv2.GaussianBlur(enhanced, (15, 15), 0)

        # 4. Otsu's Binarization
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Mask out everything outside the detected AprilTags (ROI Locking)
        # This prevents detection of aluminum rails and distorted edge artifacts.
        from .calibration import calibration_service
        tags = calibration_service.detect_tags(image)
        
        mask_roi = np.zeros_like(thresh)
        if len(tags) >= 3:
            # Create a convex hull of all tag corners to define the active workspace
            all_tag_pts = np.vstack([np.array(t['corners'], dtype=np.int32) for t in tags])
            hull = cv2.convexHull(all_tag_pts)
            cv2.fillPoly(mask_roi, [hull], 255)
        else:
            # Fallback to a safe center margin if tags aren't visible
            margin = 80
            mask_roi[margin:-margin, margin:-margin] = 255
            
        thresh = cv2.bitwise_and(thresh, mask_roi)

        # 5. Specifically mask out the tags themselves so they aren't detected as objects
        for tag in tags:
            pts = np.array(tag['corners'], dtype=np.int32)
            cv2.fillPoly(thresh, [pts], 0)

        # 6. Heavy Morphological Closing to "glue" fragments into solid coasters
        # We use a 35px kernel to bridge the dark gaps between engravings
        kernel_size = max(self.honeycomb_kernel, 35)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # Find outermost contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            # Filter out tiny noise and the massive frame around the edge of the bed
            if area_px < 1000 or area_px > (orig_w * orig_h * 0.5):
                continue

            # Build binary mask from contour
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                "_box": [x, y, x + w, y + h],
                "_mask": mask,
                "_confidence": 1.0,
                "_class_id": -1,  # software fallback — no class
            })

        return self._build_workpiece_list(detections, orig_h, orig_w)

    # ── Shared post-processing ────────────────────────────────────────────────

    def _build_workpiece_list(self, detections: List[Dict],
                               orig_h: int, orig_w: int) -> List[Dict]:
        """
        Convert raw detections into the unified workpiece schema.
        Applies physical-area filtering via the calibration homography.
        """
        results = []
        for i, det in enumerate(detections):
            mask = det["_mask"]
            x1, y1, x2, y2 = det["_box"]

            # Extract contour from mask for oriented box & segmentation polygon
            contour_pts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contour_pts:
                continue
            cnt = max(contour_pts, key=cv2.contourArea)

            # Oriented bounding rect → 4 corners + angle
            rect = cv2.minAreaRect(cnt)
            corners_px = cv2.boxPoints(rect).astype(int)  # (4, 2)
            angle_deg = float(rect[2])

            # Convex hull as segmentation polygon
            hull = cv2.convexHull(cnt)
            seg_px = hull.reshape(-1, 2).tolist()

            # Map to physical mm via calibration homography
            corners_mm = calibration_service.map_pixels_to_mm(
                corners_px.astype(np.float32)
            )
            seg_mm = calibration_service.map_pixels_to_mm(
                np.array(seg_px, dtype=np.float32)
            )

            # Physical area filter
            area_mm2 = float(cv2.contourArea(corners_mm.reshape(-1, 1, 2).astype(np.float32)))
            if area_mm2 < self.min_area_mm2 or area_mm2 > self.max_area_mm2:
                logger.debug(f"Filtered detection {i}: area {area_mm2:.0f} mm² out of range")
                continue

            box_mm = [
                float(corners_mm[:, 0].min()),
                float(corners_mm[:, 1].min()),
                float(corners_mm[:, 0].max()),
                float(corners_mm[:, 1].max()),
            ]

            results.append({
                "id": f"wp_{i:03d}",
                "class": "workpiece",
                "confidence": det["_confidence"],
                "angle_deg": angle_deg,
                "box_px": [x1, y1, x2, y2],
                "corners_px": corners_px.tolist(),
                "segmentation_px": seg_px,
                "corners_mm": corners_mm.tolist(),
                "segmentation_mm": seg_mm.tolist(),
                "box_mm": box_mm,
                "area_mm2": area_mm2,
            })

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_workpieces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect workpieces in the image.
        Returns a list of dicts with id, class, confidence, angle_deg,
        box_px, corners_px, segmentation_px, corners_mm, segmentation_mm,
        box_mm, area_mm2.
        """
        if not self.initialized or image is None:
            return []

        # Step 0: Undistort the fisheye lens before processing
        from .calibration import calibration_service
        image = calibration_service.undistort(image)

        # Prepare debug drawing
        debug_img = image.copy()

        if self.use_hailo:
            results = self._run_hailo_inference(image)
            # If NPU found nothing confident, try software fallback
            if not results:
                logger.info("Hailo NPU found no objects. Falling back to software detection...")
                results = self._software_detect(image)
        else:
            results = self._software_detect(image)

        # Draw final detections on debug image
        for wp in results:
            pts = np.array(wp['corners_px'], dtype=np.int32)
            mm_pts = wp['corners_mm']
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
            
            # Label the ID at the top
            label = f"{wp['id']}"
            cv2.putText(debug_img, label, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Label each individual corner with its physical (X, Y)
            for i, (px, mm) in enumerate(zip(pts, mm_pts)):
                mm_label = f"({int(mm[0])},{int(mm[1])})"
                cv2.putText(debug_img, mm_label, (px[0], px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Save debug image
        debug_path = "calibration_data/detect_debug.jpg"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, debug_img)

        return results


# Global instance
inference_service = InferenceService()
