# file: detection/detection.py

import cv2
import numpy as np
import onnxruntime as ort

from config.constants import (
    ONNX_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
)

# Pascal VOC PERSON = index 15
PERSON_CLASS_ID = 15

# MobileNet-SSD normalization constants
SSD_IMAGE_MEAN = np.array([127, 127, 127], dtype=np.float32)
SSD_IMAGE_STD = 128.0


class PersonDetector:
    def __init__(self, model_path: str = ONNX_MODEL_PATH,
                 conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold

        # Load ONNX MobileNet-SSD
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Input tensor metadata
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        _, self.ch, self.in_h, self.in_w = meta.shape  # expected (1, 3, 300, 300)

        # Outputs: scores [1, N, 21], boxes [1, N, 4]
        outs = self.session.get_outputs()
        self.output_scores = outs[0].name
        self.output_boxes = outs[1].name

        print(f"✓ ONNX loaded: {model_path}")
        print(f"Input shape: CHW = ({self.ch}, {self.in_h}, {self.in_w})")

        # cache for bbox restoration
        self.last_scale = 1.0
        self.last_new_w = self.in_w
        self.last_new_h = self.in_h
        self.last_W = self.in_w
        self.last_H = self.in_h

    # ============================================================
    # PREPROCESS — letterbox + normalization (SSD style)
    # ============================================================
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize while preserving aspect ratio, pad into 300×300,
        then normalize like the original PyTorch SSD predictor.
        """
        H, W = frame.shape[:2]

        # Scale so that the LONG side fits 300 (no overflow)
        long_side = max(H, W)
        scale = self.in_h / float(long_side)

        new_w = int(W * scale)
        new_h = int(H * scale)

        # Resize original frame
        resized = cv2.resize(frame, (new_w, new_h))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Pad into 300×300 (top-left aligned)
        canvas = np.zeros((self.in_h, self.in_w, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = resized

        # SSD normalization
        canvas = (canvas - SSD_IMAGE_MEAN) / SSD_IMAGE_STD

        # HWC → CHW, add batch
        chw = np.transpose(canvas, (2, 0, 1))
        blob = np.expand_dims(chw, axis=0)

        # Cache for restoring coordinates
        self.last_scale = scale
        self.last_new_w = new_w
        self.last_new_h = new_h
        self.last_W = W
        self.last_H = H

        return blob

    # ============================================================
    # RESTORE BBOX — from normalized model coords to original frame
    # ============================================================
    def restore_bbox(self, box: np.ndarray):
        """
        Take a predicted box in normalized coordinates [0–1]
        (relative to the padded resized image) and map it back
        into the original camera frame size.
        """
        x1, y1, x2, y2 = box  # normalized 0–1

        # to padded resized (in pixels)
        x1 *= self.last_new_w
        x2 *= self.last_new_w
        y1 *= self.last_new_h
        y2 *= self.last_new_h

        # undo overall scale (letterbox)
        x1 /= self.last_scale
        x2 /= self.last_scale
        y1 /= self.last_scale
        y2 /= self.last_scale

        # clip to original frame bounds
        x1 = max(0, min(self.last_W, x1))
        x2 = max(0, min(self.last_W, x2))
        y1 = max(0, min(self.last_H, y1))
        y2 = max(0, min(self.last_H, y2))

        return int(x1), int(y1), int(x2), int(y2)

    # ============================================================
    # ZONE DECISION — based on overlap with left/center/right
    # ============================================================
    @staticmethod
    def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
        """
        1D overlap length between segment [a0, a1] and [b0, b1].
        """
        left = max(a0, b0)
        right = min(a1, b1)
        return max(0, right - left)

    def _classify_zone(self, bbox, frame_width: int) -> str:
        """
        Classify LEFT / CENTER / RIGHT using overlap between the
        bounding box and each zone.
        Zones:
           LEFT   : [0, 0.35W)
           CENTER : [0.35W, 0.65W)
           RIGHT  : [0.65W, W)
        """
        x1, _, x2, _ = bbox
        box_w = max(1, x2 - x1)

        # Define zones
        left_end = int(frame_width * 0.35)
        right_start = int(frame_width * 0.65)

        left_range = (0, left_end)
        center_range = (left_end, right_start)
        right_range = (right_start, frame_width)

        # Compute overlaps
        ol_left = self._overlap_1d(x1, x2, *left_range)
        ol_center = self._overlap_1d(x1, x2, *center_range)
        ol_right = self._overlap_1d(x1, x2, *right_range)

        overlaps = {
            "LEFT": ol_left,
            "CENTER": ol_center,
            "RIGHT": ol_right,
        }

        # If overlaps are all tiny (e.g. detection way out of frame), default to CENTER
        max_ol = max(overlaps.values())
        if max_ol < box_w * 0.1:  # very little overlap anywhere
            return "CENTER"

        # Pick zone with largest overlap
        zone = max(overlaps.items(), key=lambda kv: kv[1])[0]
        return zone

    # ============================================================
    # MAIN DETECTION API
    # ============================================================
    def detect(self, frame: np.ndarray, zones=None) -> dict:
        """
        Run ONNX MobileNet-SSD on a single frame and return:
          {
            "found": bool,
            "zone": "LEFT" | "CENTER" | "RIGHT" | None,
            "bbox": (x1, y1, x2, y2) or None,
            "conf": float          # best person confidence
          }
        """
        blob = self.preprocess(frame)

        # Run ONNX inference
        scores, boxes = self.session.run(
            [self.output_scores, self.output_boxes],
            {self.input_name: blob},
        )

        scores = scores[0]  # [N, num_classes]
        boxes = boxes[0]    # [N, 4]

        # Debug: check model health
        try:
            print("Max class confidence:", float(np.max(scores)))
        except Exception:
            pass

        best_conf = 0.0
        best_bbox = None

        # Pick the single best PERSON box
        num_priors = scores.shape[0]
        for i in range(num_priors):
            conf = float(scores[i][PERSON_CLASS_ID])
            if conf < self.conf_threshold:
                continue

            if conf > best_conf:
                best_conf = conf
                best_bbox = self.restore_bbox(boxes[i])

        if best_bbox is None:
            return {
                "found": False,
                "zone": None,
                "bbox": None,
                "conf": 0.0,
            }

        # Decide LEFT / CENTER / RIGHT using overlap
        frame_w = frame.shape[1]
        zone = self._classify_zone(best_bbox, frame_w)

        return {
            "found": True,
            "zone": zone,
            "bbox": best_bbox,
            "conf": best_conf,
        }
