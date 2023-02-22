# MIT License
# 
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Any, Dict, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np

from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1


class Transmitter(NamedTuple):
    """A configuration utility to set up agents to transmit messages."""

    pass


class Receiver(NamedTuple):
    """A configuratoin utility to set up agent to receive messages."""

    pass


class MessagePasser(gym.Wrapper):
    """This wrapper augments the observations and actions to require passing messages from agents."""

    def __init__(self, env: gym.Env, message_size=125000):
        super().__init__(env)
        assert isinstance(env, HiWayEnvV1)
        o_action_space: gym.spaces.Dict = self.env.action_space
        msg_space = (
            gym.spaces.Box(low=0, high=256, shape=(message_size,), dtype=np.uint8),
        )
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

    def step(self, action):
        std_actions = {}
        msgs = {}
        for a_id, ma in action.items():
            std_actions[a_id] = ma[0]
            msgs[a_id] = ma[1]

        obs, rewards, terms, truncs, infos = self.env.step(std_actions)
        obs_with_msgs = {
            "agents": obs,
            "messages": {a_id: msgs for a_id in obs},
        }
        return obs_with_msgs, rewards, terms, truncs, infos

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return {"agents": obs, "messages": self.observation_space["messages"].sample()}
