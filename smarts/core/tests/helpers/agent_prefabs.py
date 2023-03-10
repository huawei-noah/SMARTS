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
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.observations import Observation
from smarts.zoo import AgentSpec
from smarts.zoo.registry import make, register


class KeepPoseAgent(Agent):
    def act(self, obs: Observation, **configs):
        return [*obs.ego_vehicle_state.position[:2], obs.ego_vehicle_state.heading, 0.1]


def kp_entrypoint(*args, **kwargs):
    return AgentSpec(
        interface=AgentInterface(debug=True, action=ActionSpaceType.TargetPose),
        agent_builder=KeepPoseAgent,
        agent_params=kwargs,
    )


class MoveToTargetPoseAgent(Agent):
    def __init__(self, target_pose) -> None:
        self._target_pose = target_pose

    def act(self, obs: Observation, **configs):
        return [*self._target_pose.position[:2], self._target_pose.heading, 0.1]


def mtp_entrypoint(target_pose):
    return AgentSpec(
        interface=AgentInterface(debug=True, action=ActionSpaceType.TargetPose),
        agent_builder=MoveToTargetPoseAgent,
        agent_params=[target_pose],
    )


register("keep-pose-v0", kp_entrypoint)
register("move-to-target-pose-v0", mtp_entrypoint)
