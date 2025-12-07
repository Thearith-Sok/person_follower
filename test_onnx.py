import cv2
import onnxruntime as ort
import numpy as np

# --- CONFIG ---
onnx_path = "models/person_follower.onnx"
labels_path = "models/voc-model-labels.txt"
image_path = "D:\\vscode\\pythonproject\\robotic_final\\images.jpg"  # <-- CHANGE THIS TO A REAL IMAGE PATH
# ---------------

# Load labels
class_names = [l.strip() for l in open(labels_path)]

print("Loading ONNX model...")
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

# Load image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

orig_h, orig_w = image.shape[:2]

# Preprocessing
img = cv2.resize(image, (300, 300)).astype(np.float32)
img = img[:, :, ::-1]           # BGR → RGB
img = (img - 127.0) / 128.0     # Normalize
img = np.transpose(img, (2, 0, 1))  # HWC → CHW
img = np.expand_dims(img, axis=0).astype(np.float32)

# ONNX inference
scores, boxes = session.run(None, {input_name: img})

scores = scores[0]
boxes = boxes[0]

PERSON = 15  # VOC "person" class id

for i in range(len(scores)):
    class_id = np.argmax(scores[i])
    score = scores[i][class_id]

    if class_id == PERSON and score > 0.5:
        y1, x1, y2, x2 = boxes[i]

        x1 = int(x1 * orig_w)
        x2 = int(x2 * orig_w)
        y1 = int(y1 * orig_h)
        y2 = int(y2 * orig_h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Person {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

cv2.imwrite("onnx_result.jpg", image)
print("Saved → onnx_result.jpg")
