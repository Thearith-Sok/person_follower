# file: main.py

import time

# --- Imports from your modules ---
from robot.auppbot import AUPPBot
from camera.video_stream import VideoStream
from detection.detection import PersonDetector
from decision.decision import decide_action
from actions.actions import move_forward, turn_left, turn_right, stop
from web.web_ui import WebUI

from config.constants import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, ROTATE_180,
    PORT, BAUD,
    BASE_SPEED, TURN_SPEED, STOP_SPEED
)


def main():
    print("Starting Person-Following Robot...")

    # --- Initialize Web UI ---
    web = WebUI(port=5000)

    # --- Initialize Camera ---
    stream = VideoStream(
        cam_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        rotate_180=ROTATE_180
    )

    # --- Initialize Person Detector ---
    detector = PersonDetector()

    # --- Initialize Robot Hardware ---
    bot = None
    try:
        bot = AUPPBot(PORT, BAUD, auto_safe=True)
        print("✓ Connected to AUPPBot")
    except Exception as e:
        print("⚠️ Robot not connected:", e)
        print("Running in camera-only mode.")

    # --- Main Loop ---
    while True:
        frame = stream.get_frame()
        if frame is None:
            continue

        # Split frame into 3 tracking zones
        zones = stream.split_into_zones(frame)

        # Run ONNX person detection
        detection = detector.detect(frame, zones)

        # Decide next robot action
        action = decide_action(detection)

        # --- Robot Movement Logic ---
        if bot is not None:
            if action == "MOVE_FORWARD":
                move_forward(bot, BASE_SPEED)

            elif action == "TURN_LEFT":
                turn_left(bot, TURN_SPEED)

            elif action == "TURN_RIGHT":
                turn_right(bot, TURN_SPEED)

            else:
                stop(bot)

        # --- Update Web UI (frame + zones + detection overlays) ---
        web.update_frame(frame, zones, detection)

        # Debug print
        print(f"[{action}]  Detection: {detection}")

        time.sleep(0.03)  # Stability delay


if __name__ == "__main__":
    main()
