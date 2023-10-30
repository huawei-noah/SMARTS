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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import multiprocessing
import tempfile
from pathlib import Path

import pytest
import websocket

from envision.client import Client as Envision
from envision.utils.multiprocessing_queue import Queue
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "Agent-007"
NUM_EPISODES = 3
MAX_STEPS = 100
TIMESTEP_SEC = 0.1


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


@pytest.fixture
def agent_spec():

    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=MAX_STEPS
        ),
        agent_builder=KeepLaneAgent,
    )


@pytest.fixture
def data_replay_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def scenarios_iterator():
    scenarios = ["scenarios/sumo/loop", "scenarios/sumo/intersections/6lane"]
    return Scenario.scenario_variations(scenarios, [AGENT_ID])


def fake_websocket_app_class():
    # Using a closure instead of a class field to give isolation between tests.
    sent = Queue(ctx=multiprocessing.get_context())

    class FakeWebSocketApp:
        """Mocks out the websockets.WebSocketApp to intercept send(...) calls and just
        store them locally for later evaluation.
        """

        def __init__(self, endpoint, on_error, on_close, on_open):
            self._on_error = on_error
            self._on_close = on_close
            self._on_open = on_open
            self.url = "localhost:0000"

        def send(self, data):
            sent.put(data)
            return len(data)

        def run_forever(self):
            self._on_open(self)

        def close(self):
            pass

    return FakeWebSocketApp, sent


def test_data_replay(agent_spec, scenarios_iterator, data_replay_path, monkeypatch):
    """We stub out the websocket the Envision client writes to and store the sent data.
    We do the same under Envision client's replay feature and compare that the data
    sent to the websocket is the same as before.
    """

    def step_through_episodes(agent_spec, smarts, scenarios_iterator):
        steps = 0
        for i in range(NUM_EPISODES):
            agent = agent_spec.build_agent()
            scenario = next(scenarios_iterator)
            obs = smarts.reset(scenario)

            done = False
            while not done:
                obs = obs[AGENT_ID]
                obs, _, dones, _ = smarts.step({AGENT_ID: agent.act(obs)})
                done = dones[AGENT_ID]
                steps += 1

        return steps

    # 1. Inspect sent data during SMARTS simulation

    # Mock WebSocketApp so we can inspect the websocket frames being sent
    FakeWebSocketApp, original_sent_data = fake_websocket_app_class()
    monkeypatch.setattr(websocket, "WebSocketApp", FakeWebSocketApp)
    assert original_sent_data.qsize() == 0

    envision = Envision(output_dir=data_replay_path)
    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        traffic_sims=[SumoTrafficSimulation(time_resolution=TIMESTEP_SEC)],
        envision=envision,
        fixed_timestep_sec=TIMESTEP_SEC,
    )
    steps = step_through_episodes(agent_spec, smarts, scenarios_iterator)
    assert steps > 0, "not enough steps to send data"
    smarts.destroy()

    data_replay_path = Path(data_replay_path)
    data_replay_run_paths = [x for x in data_replay_path.iterdir() if x.is_dir()]
    assert len(data_replay_run_paths) == 1

    jsonl_paths = list(data_replay_run_paths[0].glob("*.jsonl"))
    assert len(jsonl_paths) == 1
    assert original_sent_data.qsize() > 0

    # 2. Inspect replay data

    # Mock WebSocketApp so we can inspect the websocket frames being sent
    FakeWebSocketApp, new_sent_data = fake_websocket_app_class()
    monkeypatch.setattr(websocket, "WebSocketApp", FakeWebSocketApp)
    assert new_sent_data.qsize() == 0

    # Now read data replay
    Envision.read_and_send(
        str(jsonl_paths[0].absolute()), fixed_timestep_sec=TIMESTEP_SEC
    )

    # Verify the new data matches the original data
    assert original_sent_data.qsize() == new_sent_data.qsize()
    for _ in range(new_sent_data.qsize()):
        assert original_sent_data.get() == new_sent_data.get()
