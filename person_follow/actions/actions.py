# file: actions/actions.py

def move_forward(bot, speed):
    """Move robot forward at given speed."""
    bot.motor1.forward(speed)
    bot.motor2.forward(speed)
    bot.motor3.forward(speed)
    bot.motor4.forward(speed)


def move_backward(bot, speed):
    """Move robot backward."""
    bot.motor1.backward(speed)
    bot.motor2.backward(speed)
    bot.motor3.backward(speed)
    bot.motor4.backward(speed)


def turn_left(bot, speed):
    """Turn robot left (rotate on spot)."""
    bot.motor1.backward(speed)
    bot.motor2.backward(speed)
    bot.motor3.forward(speed)
    bot.motor4.forward(speed)


def turn_right(bot, speed):
    """Turn robot right (rotate on spot)."""
    bot.motor1.forward(speed)
    bot.motor2.forward(speed)
    bot.motor3.backward(speed)
    bot.motor4.backward(speed)


def stop(bot):
    """Stop all motors."""
    bot.motor1.stop()
    bot.motor2.stop()
    bot.motor3.stop()
    bot.motor4.stop()