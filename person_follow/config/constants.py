# file: config/constants.py

import os

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Adjust this if your ONNX file lives somewhere else
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "/home/aupp/person_follow/person_follower.onnx")

# Detection
CONFIDENCE_THRESHOLD = 0.35

# Camera
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Robot serial
SERIAL_PORT = "/dev/ttyUSB0"   # change to your port if needed
BAUD_RATE = 115200

# Movement parameters (tweak on real robot)
BASE_SPEED = 16          # forward speed when following
TURN_DELTA = 4           # how strong to turn left/right
SEARCH_SPIN_SPEED = 12   # speed when spinning to search for person

# "Memory" when person is briefly lost
MEMORY_SECONDS = 0.8     # keep moving based on last seen zone for this many seconds

# Debug options
DEBUG_PRINT = True
DEBUG_DRAW = False       # True if you connect a monitor and want OpenCV windows
