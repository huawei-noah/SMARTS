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
import itertools

import pytest
from helpers.scenario import temp_scenario

from smarts.core.scenario import Scenario
from smarts.core.utils.id import SocialAgentId
from smarts.sstudio import gen_missions, gen_social_agent_missions
from smarts.sstudio.types import Mission, Route, SocialAgentActor

AGENT_ID = "Agent-007"


@pytest.fixture
def scenario_root():
    # TODO: We may want to consider referencing to concrete scenarios in our tests
    #       rather than generating them. The benefit of generting however is that
    #       we can change the test criteria and scenario code in unison.
    with temp_scenario(name="cycles", map="maps/6lane.net.xml") as scenario_root:
        actors = [
            SocialAgentActor(
                name=f"non-interactive-agent-{speed}-v0",
                agent_locator="zoo.policies:non-interactive-agent-v0",
                policy_kwargs={"speed": speed},
            )
            for speed in [10, 30, 80]
        ]

        for name, (edge_start, edge_end) in [
            ("group-1", ("edge-north-NS", "edge-south-NS")),
            ("group-2", ("edge-west-WE", "edge-east-WE")),
            ("group-3", ("edge-east-EW", "edge-west-EW")),
            ("group-4", ("edge-south-SN", "edge-north-SN")),
        ]:
            route = Route(
                begin=("edge-north-NS", 1, 0), end=("edge-south-NS", 1, "max")
            )
            missions = [Mission(route=route)] * 2  # double up
            gen_social_agent_missions(
                scenario_root,
                social_agent_actor=actors,
                name=name,
                missions=missions,
            )

        gen_missions(
            scenario_root,
            missions=[
                Mission(
                    Route(begin=("edge-west-WE", 0, 0), end=("edge-east-WE", 0, "max"))
                )
            ],
        )
        yield scenario_root


def test_scenario_variations_of_social_agents(scenario_root):
    iterator = Scenario.variations_for_all_scenario_roots(
        [str(scenario_root)], [AGENT_ID]
    )
    scenarios = list(iterator)

    assert len(scenarios) == 6, "3 social agents x 2 missions each "
    for s in scenarios:
        assert len(s.social_agents) == 4, "4 social agents"
        assert len(s.missions) == 5, "4 missions for social agents + 1 for ego"

    # Ensure correct social agents are being spawned
    all_social_agent_ids = set()
    for s in scenarios:
        all_social_agent_ids |= set(s.social_agents.keys())

    groups = ["group-1", "group-2", "group-3", "group-4"]
    speeds = [10, 30, 80]
    expected_social_agent_ids = {
        SocialAgentId.new(f"non-interactive-agent-{speed}-v0", group=group)
        for group, speed in itertools.product(groups, speeds)
    }

    assert (
        len(all_social_agent_ids - expected_social_agent_ids) == 0
    ), "All the correct social agent IDs were used"
