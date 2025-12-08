#!/usr/bin/env python3
import os, time, cv2
import numpy as np
from threading import Thread, Lock
from flask import Flask, Response, jsonify, make_response
import onnxruntime as ort

# -------- config --------
MODEL_PATH = "person_follow.onnx"
CAM_INDEX  = 0
IMG_SIZE   = 300      # SSD-MobileNet default size
CONF       = 0.5       # confidence threshold

# MobileNet-SSD VOC labels
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# -------- model load --------
session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CPUExecutionProvider']
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# -------- preprocess --------
def preprocess(frame):
    # resize to 300x300
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img -= 127.5
    img /= 127.5
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img


# -------- postprocess --------
def postprocess(frame, detections):
    h, w = frame.shape[:2]

    # detections shape: [1, 1, num_det, 7]
    for det in detections[0][0]:
        conf = det[2]
        if conf < CONF:
            continue

        class_id = int(det[1])
        label = CLASSES[class_id]

        if label != "person":  # only detect person
            continue

        # bounding box
        x1 = int(det[3] * w)
        y1 = int(det[4] * h)
        x2 = int(det[5] * w)
        y2 = int(det[6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return frame


# -------- camera thread --------
class Camera:
    def __init__(self, index=0, width=None, height=None):
        self.cap = cv2.VideoCapture(index)
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ok, self.frame = self.cap.read()
        self.lock = Lock()
        self.running = True
        self.t = Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = ok, f
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()

cam = Camera(CAM_INDEX)


# -------- flask --------
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MobileNet-SSD ONNX • Raspberry Pi Stream</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body { margin:0; min-height:100vh; display:grid; place-items:center;
           background:#0b0c10; color:#eaf0f6; font-family:system-ui,Segoe UI,Roboto,sans-serif; }
    .card { width:min(96vw,900px); background:#111417; border-radius:16px; padding:14px;
            border:1px solid rgba(255,255,255,0.08); box-shadow:0 10px 40px rgba(0,0,0,.35); }
    h1 { margin:6px 0 10px; font-size:1.05rem; }
    .row { display:flex; gap:10px; justify-content:space-between; align-items:center; }
    .btn { border:1px solid rgba(255,255,255,.12); background:#1b2229; color:#eaf0f6;
           padding:6px 12px; border-radius:10px; cursor:pointer; font-weight:600; }
    .btn:hover { background:#222b33; }
    .frame { width:100%; aspect-ratio:16/9; background:#0d1117; border-radius:12px; overflow:hidden;
             border:1px solid rgba(255,255,255,0.08); display:grid; place-items:center; }
    img { width:100%; height:100%; object-fit:contain; }
    small { opacity:.65; }
  </style>
</head>
<body>
  <div class="card">
    <div class="row">
      <h1>Raspberry Pi • MobileNet-SSD (ONNX) Live</h1>
      <button class="btn" onclick="reloadStream()">Reload</button>
    </div>
    <div class="frame">
      <img id="stream" src="/stream" alt="Stream">
    </div>
    <div class="row" style="margin-top:8px;">
      <small>Status: <span id="health">checking…</span></small>
      <small>URL: <code id="url"></code></small>
    </div>
  </div>
<script>
  async function checkHealth() {
    try {
      const r = await fetch('/health', {cache:'no-store'});
      const j = await r.json();
      document.getElementById('health').textContent = j.camera_ok ? 'camera OK' : 'no camera';
    } catch (e) {
      document.getElementById('health').textContent = 'server offline';
    }
  }
  function reloadStream() {
    const img = document.getElementById('stream');
    img.src = '/stream?ts=' + Date.now();
  }
  document.getElementById('url').textContent = location.href;
  checkHealth(); setInterval(checkHealth, 4000);
</script>
</body></html>
"""


@app.route("/")
def index():
    return make_response(INDEX_HTML, 200)


@app.route("/health")
def health():
    ok, _ = cam.read()
    return jsonify({"camera_ok": bool(ok)})


def gen_mjpeg():
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        # inference
        img = preprocess(frame)
        detections = session.run([output_name], {input_name: img})
        detections = detections[0]

        # draw
        annotated = postprocess(frame, detections)

        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        b = jpg.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Cache-Control: no-cache\r\n"
               b"Content-Length: " + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")


@app.route("/stream")
def stream():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        cam.release()