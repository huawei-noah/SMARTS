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
from unittest.mock import MagicMock

import pytest
from helpers.scenario import temp_scenario

from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.sstudio import gen_social_agent_missions
from smarts.sstudio.types import Mission, Route, SocialAgentActor

AGENT_ID = "Agent-007"

traffic_history_1 = {
    "0.1": {
        "1": {
            "vehicle_id": "1",
            "vehicle_type": "car",
            "position": [1068.124, 959.455, 0],
            "speed": 5.128410474991252,
            "heading": -4.675796326794897,
            "vehicle_length": 4.22,
            "vehicle_width": 1.72,
        },
        "2": {
            "vehicle_id": "2",
            "vehicle_type": "car",
            "position": [1041.407, 956.583, 0],
            "speed": 2.9067860258367832,
            "heading": 1.4512036732051032,
            "vehicle_length": 4.16,
            "vehicle_width": 1.78,
        },
    }
}

traffic_history_2 = {
    "0.2": {
        "13": {
            "vehicle_id": "13",
            "vehicle_type": "car",
            "position": [1058.951, 950.98, 0],
            "speed": 8.8529150566353,
            "heading": 1.5092036732051035,
            "vehicle_length": 4.4,
            "vehicle_width": 1.92,
        },
    }
}


@pytest.fixture
def create_scenario():
    with temp_scenario(name="cycles", map="maps/6lane.net.xml") as scenario_root:
        actors = [
            SocialAgentActor(
                name=f"non-interactive-agent-{speed}-v0",
                agent_locator="zoo.policies:non-interactive-agent-v0",
                policy_kwargs={"speed": speed},
            )
            for speed in [10, 30]
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
            missions = [Mission(route=route)] * 2
            gen_social_agent_missions(
                scenario_root,
                social_agent_actor=actors,
                name=name,
                missions=missions,
            )

        yield scenario_root


def test_mutiple_traffic_data(create_scenario):
    Scenario.discover_traffic_histories = MagicMock(
        return_value=[traffic_history_1, traffic_history_2]
    )
    iterator = Scenario.variations_for_all_scenario_roots(
        [str(create_scenario)], [AGENT_ID], shuffle_scenarios=False
    )
    scenarios = list(iterator)

    assert len(scenarios) == 8  # 2 social agents x 2 missions x 2 histories

    traffic_history_provider = TrafficHistoryProvider()

    use_first_history = True
    for scenario in scenarios:
        if use_first_history:
            assert scenario.traffic_history is traffic_history_1
        else:
            assert scenario.traffic_history is traffic_history_2

        traffic_history_provider.setup(scenario)
        assert (
            traffic_history_provider._current_traffic_history
            == scenario.traffic_history
        )

        use_first_history = not use_first_history
