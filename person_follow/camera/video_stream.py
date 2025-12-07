# file: camera/video_stream.py

import cv2

class VideoStream:
    def __init__(self, cam_index=0, width=640, height=480, rotate_180=False):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.rotate_180 = rotate_180

        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera")

    def get_frame(self):
        """Grab a frame from the camera."""
        ok, frame = self.cap.read()
        if not ok:
            return None

        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        return frame

    def split_into_zones(self, frame):
        """
        Splits frame width into 3 equal zones:
        LEFT, CENTER, RIGHT
        Returns (left_x1, left_x2, mid_x1, mid_x2, right_x1, right_x2)
        """
        h, w = frame.shape[:2]

        left_end = w // 3
        right_start = 2 * w // 3

        return {
            "left":  (0, left_end),
            "center": (left_end, right_start),
            "right": (right_start, w),
            "width": w,
            "height": h
        }

    def stop(self):
        """Release camera safely."""
        if self.cap:
            self.cap.release()
