from smarts.core.agent import Agent
from pynput.keyboard import Key, Listener


class ManualPolicy(Agent):
    def __init__(self):
        self.steering_rate = 0.0
        self.throttle = 0.48
        self.brake = 0.0

        self.INC_THROT = 0.01
        self.INC_STEER = 0.2

        self.ACTION_MAX_THROTTLE = 0.6
        self.ACTION_MIN_THROTTLE = 0.45

        self.ACTION_MAX_BRAKE = 1.0
        self.ACTION_MIN_BRAKE = 0.0

        self.ACTION_MAX_STEERING = 1
        self.ACTION_MIN_STEERING = -1

        # initialize the keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        """To control, use the keys:
        Up to increase the throttle
        Alt to increase the brake
        Left to decrease the steering rate
        Right to increase the steering rate
        """
        if key == Key.up:
            self.throttle = min(
                self.throttle + self.INC_THROT, self.ACTION_MAX_THROTTLE
            )
        elif key == Key.alt_l:
            self.brake = min(self.brake + 5.0 * self.INC_THROT, self.ACTION_MAX_BRAKE)
        if key == Key.right:
            self.steering_rate = min(
                self.steering_rate + self.INC_STEER, self.ACTION_MAX_STEERING
            )
        elif key == Key.left:
            self.steering_rate = max(
                self.steering_rate - self.INC_STEER, self.ACTION_MIN_STEERING
            )

    def act(self, obs):
        # discounting ..
        self.throttle = max(self.throttle * 0.99, self.ACTION_MIN_THROTTLE)
        self.steering_rate = self.steering_rate * 0.3
        self.brake = self.brake * 0.99

        # send the action
        self.action = [self.throttle, self.brake, self.steering_rate]
        return self.action
