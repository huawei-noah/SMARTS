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
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "Agent-007"
MAX_EPISODES = 3


@pytest.fixture
def agent_spec():
    return AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=100),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )


@pytest.fixture
def env(agent_spec: AgentSpec):
    agent_interfaces = {AGENT_ID: agent_spec.interface}
    agent_ids = set(agent_interfaces)
    env: HiWayEnv = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/sumo/loop"],
        agent_interfaces=agent_interfaces,
        headless=True,
        fixed_timestep_sec=0.01,
    )
    assert isinstance(env, HiWayEnv)
    assert not (agent_ids - set(env.agent_interfaces))
    matching_items = [
        env.agent_interfaces[k] == agent_interfaces[k]
        for k in env.agent_interfaces
        if k in agent_interfaces
    ]
    assert all(matching_items)
    assert len(env.agent_specs) == len(agent_interfaces)
    assert not (agent_ids - env.agent_ids)
    yield env
    env.close()


def test_hiway_env(env: HiWayEnv, agent_spec: AgentSpec):
    episode = None
    for episode in episodes(n=MAX_EPISODES):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            obs = observations[AGENT_ID]
            observations, rewards, dones, infos = env.step({AGENT_ID: agent.act(obs)})

            # Reward is currently the delta in distance travelled by the agent.
            # Ensure that it is infact a delta and not total distance travelled
            # since this bug has appeared a few times. Verify by ensuring the
            # reward does not grow unbounded.
            assert all(
                [-3 < reward < 3 for reward in rewards.values()]
            ), f"Expected bounded reward per timestep, but got {rewards}."

            episode.record_step(observations, rewards, dones, infos)

    assert episode is not None and episode.index == (
        MAX_EPISODES - 1
    ), "Simulation must cycle through to the final episode."
