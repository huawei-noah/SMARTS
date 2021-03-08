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
import json
from unittest.mock import MagicMock

import pytest
from helpers.scenario import temp_scenario

from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.sstudio import gen_social_agent_missions
from smarts.sstudio.types import Mission, Route, SocialAgentActor

AGENT_ID = "Agent-007"

vehicle_data = {
    "1": {
        "vehicle_id": "1",
        "vehicle_type": "car",
        "position": [1068.124, 959.455, 0],
        "speed": 5.128410474991252,
        "heading": -4.675796326794897,
        "vehicle_length": 4.22,
        "vehicle_width": 1.72,
    },
}
traffic_history = {}


@pytest.fixture
def history_file(tmp_path):
    # generate traffic_history of 1000 length
    timestamp = 0.0
    while timestamp < 100:
        traffic_history[str(timestamp)] = vehicle_data
        timestamp = round(timestamp + 0.1, 1)

    d = tmp_path / "sub"
    d.mkdir()
    p = d / "traffic_history.json"
    p.write_text(json.dumps(traffic_history))
    return p


@pytest.fixture
def create_scenario():
    with temp_scenario(name="cycles", map="maps/6lane.net.xml") as scenario_root:
        yield scenario_root


def check_history(range_start, range_end, data):
    for timestamp in data:
        assert range_start <= float(timestamp) and float(timestamp) <= range_end
        assert data[timestamp] == traffic_history[timestamp]


def test_mutiple_traffic_data(create_scenario, history_file):
    Scenario.discover_traffic_histories = MagicMock(return_value=[history_file])
    iterator = Scenario.variations_for_all_scenario_roots(
        [str(create_scenario)], [AGENT_ID], shuffle_scenarios=False
    )
    scenario = next(iterator)

    # After Traffic_history_service init:
    traffic_history_service = scenario.traffic_history_service
    assert len(traffic_history_service.traffic_history) == 300
    check_history(0, 29.9, traffic_history_service.traffic_history)
    # child process prepared 30.0 -> 59.9
    # assert next batch range is correct for next request
    assert (
        scenario.traffic_history_service._range_start
        == 2 * traffic_history_service._batch_size
    )

    # when reached 30.0
    traffic_history_service.fetch_history_at_timestep("30.0")
    assert len(traffic_history_service.traffic_history) == 600
    assert len(traffic_history_service._current_traffic_history) == 300
    assert len(traffic_history_service._prev_batch_history) == 300
    assert (
        traffic_history_service._range_start == 3 * traffic_history_service._batch_size
    )
    check_history(0, 59.9, traffic_history_service.traffic_history)

    # when reached 60.0
    scenario.traffic_history_service.fetch_history_at_timestep("60.0")
    assert len(traffic_history_service.traffic_history) == 600
    assert len(traffic_history_service._current_traffic_history) == 300
    assert len(traffic_history_service._prev_batch_history) == 300
    assert (
        traffic_history_service._range_start == 4 * traffic_history_service._batch_size
    )
    check_history(30.0, 89.9, traffic_history_service.traffic_history)

    # when reached 90.0, there is only 100 records left in json file
    scenario.traffic_history_service.fetch_history_at_timestep("90.0")
    assert len(traffic_history_service.traffic_history) == 400
    assert len(traffic_history_service._current_traffic_history) == 100
    assert len(traffic_history_service._prev_batch_history) == 300
    assert (
        traffic_history_service._range_start == 5 * traffic_history_service._batch_size
    )
    check_history(60.0, 99.9, traffic_history_service.traffic_history)
