import gym
from typing import Any, Dict, Tuple

Action = Any
Operation = Any


class EpisodeLogger(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(env, self).__init__(env)
        self._first_step = False

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        return super().step(action)

    def reset(self) -> Any:
        return super().reset()
