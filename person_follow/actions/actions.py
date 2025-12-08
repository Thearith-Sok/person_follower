# file: actions/actions.py

from typing import Optional

from config.constants import DEBUG_PRINT
from decision.decision import MotionCommand


def _clamp_speed(value: int) -> int:
    return max(-99, min(99, int(value)))


def apply_motion_command(bot, cmd: MotionCommand):
    """
    Apply MotionCommand speeds to the AUPPBot instance.
    If bot is None, just print (dry-run mode).
    """
    left = _clamp_speed(cmd.left_speed)
    right = _clamp_speed(cmd.right_speed)

    if DEBUG_PRINT:
        print(f"[ACTION] {cmd.label:12s} | L={left:3d} R={right:3d}")

    if bot is None:
        return

    try:
        # Assuming AUPPBot API like your line-follower code
        bot.motor1.speed(left)
        bot.motor2.speed(left)
        bot.motor3.speed(right)
        bot.motor4.speed(right)
    except Exception as e:
        if DEBUG_PRINT:
            print("⚠️  Error while commanding motors:", e)


def stop_bot(bot: Optional[object]):
    """
    Safely stop robot movement and release resources.
    """
    if bot is None:
        return

    try:
        bot.motor1.speed(0)
        bot.motor2.speed(0)
        bot.motor3.speed(0)
        bot.motor4.speed(0)
    except Exception:
        pass

    try:
        bot.stop_all()
    except Exception:
        pass

    try:
        bot.close()
    except Exception:
        pass
