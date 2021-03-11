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
import gym
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

AGENT_ID = "Agent-007"
SOCIAL_AGENT_ID = "Alec Trevelyan"

MAX_EPISODES = 3


@pytest.fixture
def agent_spec():
    return AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=100),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )


@pytest.fixture
def env(agent_spec):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/zoo_intersection"],
        agent_specs={AGENT_ID: agent_spec},
        headless=True,
        visdom=False,
        timestep_sec=0.01,
    )

    yield env
    env.close()


def test_social_agents(env, agent_spec):
    for episode in episodes(n=MAX_EPISODES):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            obs = observations[AGENT_ID]
            observations, rewards, dones, infos = env.step({AGENT_ID: agent.act(obs)})
            episode.record_step(observations, rewards, dones, infos)

            assert SOCIAL_AGENT_ID not in observations
            assert SOCIAL_AGENT_ID not in dones

            # Reward is currently the delta in distance travelled by this agent.
            # We want to make sure that this is infact a delta and not total distance
            # travelled since this bug has appeared a few times.
            #
            # The way to verify this is by making sure the reward does not grow without bounds
            assert -3 < rewards[AGENT_ID] < 3

    assert episode.index == (
        MAX_EPISODES - 1
    ), "Simulation must cycle through to the final episode"
