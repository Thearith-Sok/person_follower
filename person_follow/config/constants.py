# file: config/constants.py

# ========== CAMERA CONFIG ==========
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ROTATE_180 = False   # set True if camera is upside-down

# ========== PERSON DETECTION CONFIG ==========
ONNX_MODEL_PATH = "models/person_ssd.onnx"  # Change to your actual ONNX file
CONFIDENCE_THRESHOLD = 0.7

# ========== ROBOT MOVEMENT ==========
BASE_SPEED = 13
TURN_SPEED = 10
STOP_SPEED = 0

# ========== SERIAL PORT CONFIG ==========
PORT = "/dev/ttyUSB0"  # Update to match your setup
BAUD = 115200

# ========== ZONE SPLITTING ==========
# Frame is split into: | LEFT | CENTER | RIGHT |
LEFT_ZONE_RATIO = 1/3
CENTER_ZONE_RATIO = 1/3
RIGHT_ZONE_RATIO = 1/3
