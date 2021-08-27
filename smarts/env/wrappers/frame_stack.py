# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import gym
from collections import deque
from typing import Dict, List, Tuple, Union
from smarts.core import sensors


class FrameStack(gym.Wrapper):
    """Wrapper stacks num_stack (default=3) consecutive frames, in a moving-window fashion, and returns the stacked_frames.

    Note:
        Wrapper returns a deepcopy of the stacked frames, which may be expensive for large frames and large num_stack values.
    """

    def __init__(self, env: gym.Env, num_stack: int = 3):
        assert num_stack > 1, f"Expected num_stack > 1, but got {num_stack}."
        super(FrameStack, self).__init__(env)
        self._num_stack = num_stack
        self._frames = {
            key: deque(maxlen=self._num_stack) for key in self.env.agent_specs.keys()
        }

    def _get_observations(
        self, frame: sensors.Observation
    ) -> Dict[str, List[sensors.Observation]]:
        """Update and return frames stack with given latest single frame."""

        new_frames = dict.fromkeys(frame)

        for agent_id, observation in frame.items():
            self._frames[agent_id].append(observation)
            frames_list = list(self._frames[agent_id])
            new_frames[agent_id] = copy.deepcopy(frames_list)

        return new_frames

    def step(
        self, agent_actions: Dict
    ) -> Tuple[
        Dict[str, List[sensors.Observation]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Union[float, sensors.Observation]]],
    ]:

        env_observations, rewards, dones, infos = super(FrameStack, self).step(
            agent_actions
        )

        return self._get_observations(env_observations), rewards, dones, infos

    def reset(self) -> Dict[str, List[sensors.Observation]]:
        env_observations = super(FrameStack, self).reset()
        for agent_id, observation in env_observations.items():
            for _ in range(self._num_stack - 1):
                self._frames[agent_id].append(observation)

        return self._get_observations(env_observations)
