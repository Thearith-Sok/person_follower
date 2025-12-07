# file: web/web_ui.py

import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, render_template_string

# ----- Shared frame for MJPEG stream -----
latest_frame = None
frame_lock = threading.Lock()

# ==========================
#   HTML (Dark Mode UI)
# ==========================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Camera View</title>
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 {
            color: #76c7ff;
            margin-bottom: 10px;
        }
        .camera-box {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        img {
            border: 3px solid #333;
            border-radius: 10px;
        }
    </style>
</head>
<body>

    <h1>🤖 Person Follower Robot — Live Camera</h1>

    <div class="camera-box">
        <img src="/video_feed" width="640" height="480">
    </div>

</body>
</html>
"""

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


def encode_jpeg(img):
    ret, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ret else None


@app.route("/video_feed")
def video_feed():
    return Response(_frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def _frame_generator():
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for camera...",
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
                frame = blank
            else:
                frame = latest_frame.copy()

        jpeg = encode_jpeg(frame)
        if jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg + b"\r\n")

        time.sleep(0.03)


# ===========================================================
# Public API (called from main.py)
#   update_frame(frame, zones, detection)
# ===========================================================
class WebUI:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port

        # Start Flask server in background thread
        thread = threading.Thread(
            target=lambda: app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            ),
            daemon=True
        )
        thread.start()

        print(f"🌐 Web UI running at http://{self.host}:{self.port}")

    def update_frame(self, frame, zones=None, detection=None):
        """
        frame: raw BGR frame from camera
        zones: output of VideoStream.split_into_zones(frame)
        detection: dict from PersonDetector.detect()
            {
                "found": True/False,
                "zone": LEFT/CENTER/RIGHT,
                "bbox": (x1, y1, x2, y2)
            }
        """

        global latest_frame
        vis = frame.copy()   # NEVER modify the original frame

        h, w = vis.shape[:2]

        # -----------------------
        # Draw Zone Split Lines
        # -----------------------
        if zones:
            left_end = zones["left"][1]
            right_start = zones["center"][1]

            cv2.line(vis, (left_end, 0), (left_end, h), (0, 255, 255), 2)
            cv2.line(vis, (right_start, 0), (right_start, h), (0, 255, 255), 2)

        # -----------------------
        # Draw bounding box
        # -----------------------
        if detection and detection.get("found"):
            (x1, y1, x2, y2) = detection["bbox"]

            # Bounding box in green
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Person center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(vis, (center_x, center_y), 6, (0, 255, 255), -1)

            # Zone label text
            zone = detection.get("zone", "-")
            cv2.putText(vis, f"Zone: {zone}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        # store for streaming
        with frame_lock:
            latest_frame = vis