from typing import Any, Dict, Tuple

import gym


class Info(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(Info, self).__init__(env)

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment. A new "is_success" key is added to the
        returned `info` of each agent.

        Args:
            action (Dict[str, Any]): Action for each agent.

        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, done, and info, for each agent is returned.
        """
        obs, reward, done, info = self.env.step(action)

        for agent_id in info.keys():
            info[agent_id]["is_success"] = bool(info[agent_id].get("score", True))

        return obs, reward, done, info
