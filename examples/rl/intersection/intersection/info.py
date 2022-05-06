from typing import Any, Dict, Tuple

import gym


class Info(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(Info, self).__init__(env)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Steps the environment. A new "is_success" key is added to the
        returned `info`.

        Args:
            action (Any): Action for the agent.

        Returns:
            Tuple[ Any, float, bool, Dict[str, Any] ]:
                Observation, reward, done, and info, for the agent is returned.
        """
        obs, reward, done, info = self.env.step(action)
        info["is_success"] = bool(info["score"])

        return obs, reward, done, info
