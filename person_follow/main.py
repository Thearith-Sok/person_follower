# file: main.py

#!/usr/bin/env python3
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # prevent Qt errors on headless Pi

import time
import cv2

from robot.auppbot import AUPPBot
from camera.video_stream import VideoStream
from detection.detection import PersonDetector
from decision.decision import PersonFollowerBrain
from actions.actions import apply_motion_command, stop_bot

from config.constants import (
    SERIAL_PORT,
    BAUD_RATE,
    DEBUG_DRAW,
    DEBUG_PRINT,
)


# ------------------------------------------------------------
# OPTIONAL: Debug overlay
# ------------------------------------------------------------
def draw_debug(frame, detection: dict, cmd_label: str):
    h, w = frame.shape[:2]

    if detection.get("found", False):
        bbox = detection.get("bbox")
        conf = detection.get("conf", 0.0)
        zone = detection.get("zone", "")

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{zone} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

    cv2.putText(
        frame, f"CMD: {cmd_label}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )

    return frame


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    bot = None
    stream = None

    try:
        # ------------------ ROBOT INIT ------------------------
        try:
            bot = AUPPBot(SERIAL_PORT, BAUD_RATE, auto_safe=True)
            print(f"✓ AUPPBot connected on {SERIAL_PORT}")
        except Exception as e:
            print("⚠️ Robot connection failed — running in dry mode:", e)
            bot = None

        # ------------------ CAMERA INIT ------------------------
        stream = VideoStream().start()
        time.sleep(0.4)
        print("✓ Video stream started")

        # ------------------ DETECTOR + BRAIN -------------------
        detector = PersonDetector()
        brain = PersonFollowerBrain()
        print("✅ Person follower optimized runtime started.\n")

        # --------------------------------------------------------
        # PERFORMANCE OPTIMIZATION SETTINGS
        # --------------------------------------------------------
        DETECT_EVERY_N_FRAMES = 3    # ONNX runs every 3 frames → ~3× FPS boost
        LOOP_SLEEP = 0.01            # keep-alive → prevents robot timeout
        last_cmd_label = None        # throttle repeated commands
        frame_id = 0

        # FPS stats
        fps_time = time.time()
        frame_counter = 0

        # ----------------------- MAIN LOOP ----------------------
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_id += 1
            frame_counter += 1

            # ------------------ DETECTION SKIPPING -------------------
            if frame_id % DETECT_EVERY_N_FRAMES == 0:
                detection = detector.detect(frame)
            else:
                # fake detection: tells brain "nothing new"
                detection = {"found": False, "zone": None, "bbox": None, "conf": 0.0}

            # ------------------ DECISION MAKING -----------------------
            cmd = brain.update(detection)

            # ------------------ COMMAND THROTTLING --------------------
            if cmd.label != last_cmd_label:
                apply_motion_command(bot, cmd)
                last_cmd_label = cmd.label

            # Always sleep briefly so serial stays alive
            time.sleep(LOOP_SLEEP)

            # ------------------ OPTIONAL VISUALIZATION -----------------
            if DEBUG_DRAW:
                vis = draw_debug(frame.copy(), detection, cmd.label)
                cv2.imshow("Person Follower Debug", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            # ------------------ FPS PRINTING ---------------------------
            now = time.time()
            if now - fps_time >= 1.0:
                fps = frame_counter / (now - fps_time)
                fps_time = now
                frame_counter = 0

                if DEBUG_PRINT:
                    print(f"[FPS] {fps:.1f}")

    # ------------------------------------------------------------
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    finally:
        # ------------------ CLEANUP ------------------------------
        stop_bot(bot)
        if stream:
            stream.stop()

        if DEBUG_DRAW:
            try: cv2.destroyAllWindows()
            except: pass

        print("✅ Cleanup complete")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
