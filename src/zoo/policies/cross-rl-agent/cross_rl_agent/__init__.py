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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# The author of this file is: https://github.com/mg2015started

import importlib.resources as pkg_resources

import cross_rl_agent

from smarts.core.agent import AgentSpec
from smarts.zoo.registry import register

from .agent import RLAgent
from .cross_space import (
    action_adapter,
    cross_interface,
    get_aux_info,
    observation_adapter,
    reward_adapter,
)


def entrypoint():
    with pkg_resources.path(cross_rl_agent, "models") as model_path:
        return AgentSpec(
            interface=cross_interface,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
            agent_builder=lambda: RLAgent(
                load_path=str(model_path) + "/",
                policy_name="Soc_Mt_TD3Network",
            ),
        )


register(locator="cross_rl_agent-v1", entry_point=entrypoint)
