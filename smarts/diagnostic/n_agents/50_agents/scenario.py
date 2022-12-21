# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, RandomRoute, Scenario, SocialAgentActor

actors = [
    SocialAgentActor(
        name=f"non-interactive-agent-{speed}-v0",
        agent_locator="zoo.policies:non-interactive-agent-v0",
        policy_kwargs={"speed": speed},
    )
    for speed in [10, 30, 80]
]


def to_missions(agent_num):
    missions = {}
    for i in range(0, agent_num):
        missions[f"group-{i}"] = tuple(
            (actors, [Mission(route=RandomRoute())]),
        )
    return missions


gen_scenario(
    Scenario(social_agent_missions=to_missions(50)),
    output_dir=Path(__file__).parent,
)
