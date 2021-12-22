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

import pytest
import time
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from helpers.scenario import temp_scenario
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.agent_interface import ActionSpaceType, AgentInterface

AGENT_1 = "Agent_007"


@pytest.fixture
def scenarios():
    with temp_scenario(name="6lane", map="maps/6lane.net.xml") as scenario_root:
        actors = [
            t.SocialAgentActor(
                name=f"non-interactive-agent-{speed}-v0",
                agent_locator="zoo.policies:non-interactive-agent-v0",
                policy_kwargs={"speed": speed},
            )
            for speed in [10, 30, 80]
        ]

        def to_mission(start_edge, end_edge):
            route = t.Route(begin=(start_edge, 1, 0), end=(end_edge, 1, "max"))
            return t.Mission(route=route)

        def fifth_mission(start_edge, end_edge):
            route = t.Route(begin=(start_edge, 0, 0), end=(end_edge, 0, "max"))
            return t.Mission(route=route)

        gen_scenario(
            t.Scenario(
                social_agent_missions={
                    "group-1": (actors, [to_mission("edge-north-NS", "edge-south-NS")]),
                    "group-2": (actors, [to_mission("edge-west-WE", "edge-east-WE")]),
                    "group-3": (actors, [to_mission("edge-east-EW", "edge-west-EW")]),
                    "group-4": (actors, [to_mission("edge-south-SN", "edge-north-SN")]),
                    "group-5": (
                        actors,
                        [fifth_mission("edge-south-SN", "edge-east-WE")],
                    ),
                },
                ego_missions=[
                    t.Mission(
                        t.Route(
                            begin=("edge-west-WE", 0, 0), end=("edge-east-WE", 0, "max")
                        )
                    )
                ],
            ),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_1]
        )


@pytest.fixture
def smarts():
    laner = AgentInterface(
        max_episode_steps=1000,
        action=ActionSpaceType.Lane,
    )

    agents = {AGENT_1: laner}
    smarts = SMARTS(
        agents,
        traffic_sim=SumoTrafficSimulation(headless=True),
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_smarts_framerate(smarts, scenarios):
    scenario = next(scenarios)
    smarts.reset(scenario)

    for _ in range(10):
        step_start_time = int(time.time() * 1000)
        smarts.step({AGENT_1: "keep_lane"})
        step_end_time = int(time.time() * 1000)
        delta = step_end_time - step_start_time
        step_fps = round(1000 / delta, 2)
        assert step_fps >= 2
