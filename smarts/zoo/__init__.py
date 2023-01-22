# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import math
import random

from smarts.core.agent import Agent
from smarts.core.agent_interface import ActionSpaceType, AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


class RandomRelativeTargetPoseAgent(Agent):
    """A simple agent that can move a random distance."""

    def __init__(self, speed=28, timestep=0.1) -> None:
        super().__init__()
        self._speed_per_step = speed / timestep

    def act(self, obs, **configs):
        return [
            (random.random() - 0.5) * self._speed_per_step,
            (random.random() - 0.5) * self._speed_per_step,
            random.random() * 2 * math.pi - math.pi,
        ]


# Note `speed` from configuration file maps here.
def entry_point(speed=10, **kwargs):
    """An example entrypoint for a simple agent.
    This can have any number of arguments similar to the gym environment standard.
    """
    return AgentSpec(
        AgentInterface(
            action=ActionSpaceType.RelativeTargetPose,
        ),
        agent_builder=RandomRelativeTargetPoseAgent,
        agent_params=dict(speed=speed),
    )


# Where the agent is registered.
register("random-relative-target-pose-agent-v0", entry_point)
