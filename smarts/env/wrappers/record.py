import gym
from typing import Any, Dict, Tuple

Action = Any
Operation = Any


class RecordVideo(gym.Wrapper):
    def __init__(self, env: gym.Env, frequency: int):
        super(RecordVideo, self).__init__(env)
        self._frequency = frequency

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        return super().step(action)

    def reset(self) -> Any:
        return super().reset()


class RenderVideo(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(RenderVideo, self).__init__(env)
        self._first_step = False

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        return super().step(action)

    def reset(self) -> Any:
        return super().reset()

    def close(self) -> None:
        return super().close()
