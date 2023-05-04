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
import gym
import pytest

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.hiway_env import HiWayEnv

AGENT_ID = "Agent-007"
SOCIAL_AGENT_ID = "Alec Trevelyan"

MAX_EPISODES = 1


@pytest.fixture
def agent_interface():
    return AgentInterface.from_type(
        AgentType.Laner, max_episode_steps=100, neighborhood_vehicle_states=True
    )


@pytest.fixture
def env(agent_interface: AgentInterface):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/sumo/zoo_intersection"],
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
        fixed_timestep_sec=0.01,
    )

    yield env
    env.close()


def test_social_agents_not_in_env_obs_keys(env: HiWayEnv):
    for _ in range(MAX_EPISODES):
        observations = env.reset()

        dones = {"__all__": False}
        while not dones["__all__"]:
            observations, rewards, dones, infos = env.step({AGENT_ID: "keep_lane"})

            assert SOCIAL_AGENT_ID not in observations
            assert SOCIAL_AGENT_ID not in dones


def test_social_agents_in_env_neighborhood_vehicle_obs(
    env: HiWayEnv, agent_interface: AgentInterface
):
    first_seen_vehicles = {}
    for _ in range(MAX_EPISODES):
        observations = env.reset()

        dones = {"__all__": False}
        while not dones["__all__"]:
            observations, rewards, dones, infos = env.step({AGENT_ID: "keep_lane"})

            new_nvs_ids = [
                nvs.id
                for nvs in observations[AGENT_ID].neighborhood_vehicle_states
                if nvs.id not in first_seen_vehicles
            ]
            for v_id in new_nvs_ids:
                first_seen_vehicles[v_id] = observations[AGENT_ID].step_count + 1

    seen_zoo_social_vehicles = [v_id for v_id in first_seen_vehicles if "zoo" in v_id]
    assert len(seen_zoo_social_vehicles) == 2
    late_entry = next(
        (v_id for v_id in seen_zoo_social_vehicles if "zoo-car1" in v_id), None
    )
    assert late_entry is not None, seen_zoo_social_vehicles
    assert first_seen_vehicles[late_entry] == 70
