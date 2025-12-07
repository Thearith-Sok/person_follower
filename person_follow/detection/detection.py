# file: detection/detection.py

import cv2
import numpy as np
import onnxruntime as ort

from config.constants import (
    ONNX_MODEL_PATH,
    CONFIDENCE_THRESHOLD
)


class PersonDetector:
    def __init__(self, model_path=ONNX_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        # Input info
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        _, self.input_ch, self.input_h, self.input_w = input_meta.shape

    # --------------------
    # Preprocessing
    # --------------------
    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)    # Add batch dimension
        return img

    # --------------------
    # Main detection call
    # --------------------
    def detect(self, frame, zone_info=None):
        """
        frame: full image from camera
        zone_info: output of VideoStream.split_into_zones(frame)
        """

        h, w = frame.shape[:2]
        blob = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})
        detections = outputs[0]  # SSD output: [1, 1, N, 7]

        best_conf = 0
        best_box = None

        # Loop through detections
        for det in detections[0][0]:
            _, class_id, conf, x1, y1, x2, y2 = det

            # MobileNet-SSD PERSON class = 15 (Pascal VOC)
            if int(class_id) != 15:
                continue

            if conf < self.conf_threshold:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = (
                    int(x1 * w),
                    int(y1 * h),
                    int(x2 * w),
                    int(y2 * h)
                )

        # No person found
        if best_box is None:
            return {
                "found": False,
                "zone": None,
                "bbox": None
            }

        x1, y1, x2, y2 = best_box
        center_x = (x1 + x2) // 2

        # -----------------------------
        # Determine LEFT / CENTER / RIGHT
        # -----------------------------
        if zone_info is None:
            # fallback: split into equal zones
            left_end = w // 3
            right_start = 2 * w // 3
        else:
            left_end = zone_info["left"][1]
            right_start = zone_info["center"][1]

        if center_x < left_end:
            zone = "LEFT"
        elif center_x < right_start:
            zone = "CENTER"
        else:
            zone = "RIGHT"

        return {
            "found": True,
            "zone": zone,
            "bbox": best_box
        }