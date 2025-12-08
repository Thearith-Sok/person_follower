# file: camera/video_stream.py

import cv2
import threading
import time

from config.constants import CAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class VideoStream:
    """
    Simple threaded camera grabber.
    Call .start(), then .read() to get the latest frame.
    """

    def __init__(self,
                 src: int = CAM_INDEX,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT):
        self.src = src
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.src}")

        # Try to set resolution (may be ignored by some webcams)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.grabbed, self.frame = self.cap.read()
        if not self.grabbed:
            raise RuntimeError("Failed to grab initial frame from camera")

        self.stopped = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                # avoid tight loop if camera disconnects
                time.sleep(0.01)
                continue

            with self._lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self._lock:
            if not self.grabbed:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
