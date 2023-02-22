import gymnasium as gym
from typing import Any, Dict, NamedTuple, Optional, Tuple
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1

import numpy as np


class Transmitter(NamedTuple):
    pass


class Receiver(NamedTuple):
    pass


class MessagePasser(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env, HiWayEnvV1)
        o_action_space: gym.spaces.Dict = self.env.action_space
        msg_space = (gym.spaces.Box(low=0, high=256, shape=(125000,), dtype=np.uint8),)
        self.action_space = gym.spaces.Dict(
            {
                a_id: gym.spaces.Tuple(
                    (
                        action_space,
                        msg_space,
                    )
                )
                for a_id, action_space in o_action_space.spaces.items()
            }
        )
        o_observation_space: gym.spaces.Dict = self.env.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                "agents": o_observation_space,
                "messages": gym.spaces.Dict(
                    {a_id: msg_space for a_id in o_action_space}
                ),
            }
        )

    def step(self, actions):
        std_actions = {}
        msgs = {}
        for a_id, ma in actions.items():
            std_actions[a_id] = ma[0]
            msgs[a_id] = ma[1]

        obs, rewards, terms, truncs, infos = self.env.step(std_actions)
        obs_with_msgs = {
            "agents": obs,
            "messages": {a_id: msgs for a_id, ob in obs.items()},
        }
        return obs_with_msgs, rewards, terms, truncs, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return {"agents": obs, "messages": self.observation_space["messages"].sample()}
