# file: decision/decision.py

import time
from collections import deque
from dataclasses import dataclass

from config.constants import (
    BASE_SPEED,
    TURN_DELTA,
    SEARCH_SPIN_SPEED,
    MEMORY_SECONDS,
)


@dataclass
class MotionCommand:
    left_speed: int
    right_speed: int
    label: str


class PersonFollowerBrain:
    """
    Person-following decision system with:
      - zone smoothing (majority vote)
      - flicker protection (0.2s zone consistency required)
      - memory behavior (continue last action briefly when target lost)
      - search behavior (spin)
    """

    def __init__(self):
        self.last_seen_time = 0.0
        self.last_seen_zone = None  # "LEFT", "CENTER", "RIGHT"
        self.zone_history = deque(maxlen=5)  # smoothing buffer
        self.last_zone_switch_time = 0.0

    # ------------------------------------------------------------
    # INTERNAL: Stabilize the zone using history + flicker filter
    # ------------------------------------------------------------
    def _get_stable_zone(self, raw_zone: str, now: float) -> str:
        """
        Take raw detection zone, apply smoothing + flicker protection,
        and return a stable zone direction.
        """

        # Add raw zone to history buffer
        self.zone_history.append(raw_zone)

        # Find zone with majority vote
        stable_zone = max(set(self.zone_history), key=self.zone_history.count)

        # Flicker protection:
        # Only allow switching zones if at least 0.2 sec since last change.
        if self.last_seen_zone is not None and stable_zone != self.last_seen_zone:
            if now - self.last_zone_switch_time < 0.2:
                # Reject the switch — keep previous zone
                stable_zone = self.last_seen_zone
            else:
                # Accept the switch
                self.last_zone_switch_time = now

        return stable_zone

    # ------------------------------------------------------------
    # MAIN: Update decision based on detection
    # ------------------------------------------------------------
    def update(self, detection: dict) -> MotionCommand:
        now = time.time()

        found = detection.get("found", False)
        raw_zone = detection.get("zone", None)

        # ============================
        # PERSON FOUND
        # ============================
        if found and raw_zone is not None:

            stable_zone = self._get_stable_zone(raw_zone, now)

            # Update memory state
            self.last_seen_time = now
            self.last_seen_zone = stable_zone

            # Movement logic
            if stable_zone == "CENTER":
                return MotionCommand(
                    left_speed=BASE_SPEED,
                    right_speed=BASE_SPEED,
                    label="follow_center",
                )

            elif stable_zone == "LEFT":
                return MotionCommand(
                    left_speed=BASE_SPEED - TURN_DELTA,
                    right_speed=BASE_SPEED + TURN_DELTA,
                    label="follow_left",
                )

            elif stable_zone == "RIGHT":
                return MotionCommand(
                    left_speed=BASE_SPEED + TURN_DELTA,
                    right_speed=BASE_SPEED - TURN_DELTA,
                    label="follow_right",
                )

            else:
                # Unexpected zone string
                return MotionCommand(0, 0, "zone_error")

        # ============================
        # PERSON NOT FOUND
        # ============================
        time_since_seen = now - self.last_seen_time

        # MEMORY: Continue last known direction briefly
        if self.last_seen_zone is not None and time_since_seen < MEMORY_SECONDS:

            if self.last_seen_zone == "LEFT":
                return MotionCommand(
                    left_speed=BASE_SPEED - TURN_DELTA,
                    right_speed=BASE_SPEED + TURN_DELTA,
                    label="memory_left",
                )

            elif self.last_seen_zone == "RIGHT":
                return MotionCommand(
                    left_speed=BASE_SPEED + TURN_DELTA,
                    right_speed=BASE_SPEED - TURN_DELTA,
                    label="memory_right",
                )

            else:
                return MotionCommand(
                    left_speed=int(BASE_SPEED * 0.7),
                    right_speed=int(BASE_SPEED * 0.7),
                    label="memory_center",
                )

        # SEARCH: spin in place if fully lost
        return MotionCommand(
            left_speed=SEARCH_SPIN_SPEED,
            right_speed=-SEARCH_SPIN_SPEED,
            label="search_spin",
        )
