# file: decision/decision.py

def decide_action(detection):
    """
    Takes the detection result dictionary:
    {
        "found": True/False,
        "zone": "LEFT" / "CENTER" / "RIGHT" / None
    }

    Returns one of the following action strings:
        - "MOVE_FORWARD"
        - "TURN_LEFT"
        - "TURN_RIGHT"
        - "STOP"
    """

    if not detection["found"]:
        return "STOP"

    zone = detection["zone"]

    if zone == "CENTER":
        return "MOVE_FORWARD"

    elif zone == "LEFT":
        return "TURN_LEFT"

    elif zone == "RIGHT":
        return "TURN_RIGHT"

    return "STOP"