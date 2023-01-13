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

from .agent_spec import AgentSpec
from .registry import register


class TestAgent(Agent):
    def act(self, obs, **configs):
        return [
            random.random() - 0.5,
            random.random() - 0.5,
            random.random() * 2 * math.pi - math.pi,
        ]


def entry_point(speed=10, **kwargs):
    return AgentSpec(
        AgentInterface(
            action=ActionSpaceType.RelativeTargetPose,
        ),
        agent_builder=TestAgent,
    )


register("competition-baseline-v0", entry_point)
