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
from smarts.zoo.registry import register
from .sac.sac.policy import SACPolicy
from .ppo.ppo.policy import PPOPolicy
from .dqn.dqn.policy import DQNPolicy
from .ddpg.ddpg.policy import TD3Policy
from .bdqn.bdqn.policy import BehavioralDQNPolicy
from smarts.core.controllers import ActionSpaceType
from ultra.baselines.agent_spec import BaselineAgentSpec

register(
    locator="sac-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=SACPolicy, **kwargs
    ),
)
register(
    locator="ppo-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=PPOPolicy, **kwargs
    ),
)
register(
    locator="ddpg-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=TD3Policy, **kwargs
    ),
)
register(
    locator="dqn-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=DQNPolicy, **kwargs
    ),
)
register(
    locator="bdqn-v0",
    entry_point=lambda **kwargs: BaselineAgentSpec(
        action_type=ActionSpaceType.Lane, policy_class=BehavioralDQNPolicy, **kwargs
    ),
)
