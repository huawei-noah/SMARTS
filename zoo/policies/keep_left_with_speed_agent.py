from smarts.core.agent import Agent


class KeepLeftWithSpeedAgent(Agent):
    def __init__(self, speed) -> None:
        self._speed = speed

    def act(self, obs):
        return self._speed, 0
