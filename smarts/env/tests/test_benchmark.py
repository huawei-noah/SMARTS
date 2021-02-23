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
import gym
import pytest

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType


@pytest.fixture(params=[5, 10])
def agent_ids(request):
    return ["AGENT-{}".format(i) for i in range(request.param)]


@pytest.fixture(params=["laner", "rgb", "lidar"])
def agent_interface(request):
    if request.param == "laner":
        return AgentInterface.from_type(AgentType.Laner)
    if request.param == "rgb":
        return AgentInterface(rgb=True, action=ActionSpaceType.Lane)
    if request.param == "lidar":
        return AgentInterface(lidar=True, action=ActionSpaceType.Lane)


@pytest.fixture
def agent_specs(agent_ids, agent_interface):
    return {id_: AgentSpec(interface=agent_interface) for id_ in agent_ids}


@pytest.fixture
def env(agent_specs):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/loop"],
        agent_specs=agent_specs,
        headless=True,
        seed=2008,
        timestep_sec=0.01,
    )
    env.reset()
    yield env
    env.close()


@pytest.mark.benchmark(group="env.step")
def test_benchmark_step(agent_ids, env, benchmark):
    @benchmark
    def env_step(counts=5):
        while counts:
            env.step({id_: "keep_lane" for id_ in agent_ids})
            counts -= 1
