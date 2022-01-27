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
from smarts.core.agent import Agent, AgentSpec


def test_building_agent_with_list_or_tuple_params():
    agent_spec = AgentSpec(
        agent_params=[32, 41],
        agent_builder=lambda x, y: Agent.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_tuple_params():
    agent_spec = AgentSpec(
        agent_params=(32, 41),
        agent_builder=lambda x, y: Agent.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_dict_params():
    agent_spec = AgentSpec(
        agent_params={"y": 2, "x": 1},
        agent_builder=lambda x, y: Agent.from_function(lambda _: x / y),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == 1 / 2
