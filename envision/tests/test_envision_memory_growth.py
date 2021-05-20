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

import tempfile
from pathlib import Path

import pytest
from pympler import muppy, summary, tracker
from envision.client import Client as Envision
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
import gc

AGENT_ID = "Agent-007"
NUM_EPISODES = 3
MAX_STEPS = 100
TIMESTEP_SEC = 0.1
EPISODE_MEMORY_GROWTH_LIMIT = 2e5

@pytest.fixture
def agent_spec():
    class KeepLaneAgent(Agent):
        def act(self, obs):
            return "keep_lane"

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
    scenarios = ["scenarios/loop", "scenarios/intersections/6lane"]
    return Scenario.scenario_variations(scenarios, [AGENT_ID])


def test_envision_memory_growth(agent_spec, scenarios_iterator, data_replay_path):
    """We test the memory growth due to emitting to envision every episode.
    """
    envision = Envision(output_dir=data_replay_path)
    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        traffic_sim=SumoTrafficSimulation(time_resolution=TIMESTEP_SEC),
        envision=envision,
        timestep_sec=TIMESTEP_SEC,
    )
    for i in range(NUM_EPISODES):
        gc.collect()
        initial_size = muppy.get_size(muppy.get_objects())
        agent = agent_spec.build_agent()
        scenario = next(scenarios_iterator)
        obs = smarts.reset(scenario)

        done = False
        while not done:
            obs = agent_spec.observation_adapter(obs[AGENT_ID])
            action = agent.act(obs)
            action = agent_spec.action_adapter(action)
            obs, _, dones, _ = smarts.step({AGENT_ID: action})
            print("episode " + str(i) + "memory_val:")
            print(muppy.get_size(muppy.get_objects()))
            done = dones[AGENT_ID]
        gc.collect()
        end_size = muppy.get_size(muppy.get_objects())

        print("episode " + str(i) + "memory_increase:")
        print(end_size - initial_size)
        assert (
                end_size - initial_size < EPISODE_MEMORY_GROWTH_LIMIT
        ), f"End size delta {end_size - initial_size}"
    smarts.destroy()
