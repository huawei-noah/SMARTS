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

import logging

import gym
import time
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)
test_logger = logging.getLogger()

AGENT_ID = "Agent-007"


@pytest.fixture
def scenarios():
    return ["scenarios/tests/multi_agents_loop"]


@pytest.fixture
def seed():
    return 42


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def env_and_spec(scenarios, seed, headless=True, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=KeepLaneAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=None,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
    )
    return env, agent_spec


def test_env_frame_test(scenarios, seed):
    env, agent_spec = env_and_spec(scenarios, seed)
    episode_counter = 0
    for episode in episodes(n=10):
        episode_counter += 1
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}

        maximum_frame_rate = 0
        minimum_frame_rate = float("inf")
        step_counter = 0
        fps_sum = 0

        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            step_start_time = int(time.time() * 1000)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            step_end_time = int(time.time() * 1000)
            delta = step_end_time - step_start_time
            step_fps = round(1000 / delta, 2)
            maximum_frame_rate = max(maximum_frame_rate, step_fps)
            minimum_frame_rate = min(minimum_frame_rate, step_fps)
            fps_sum += step_fps
            test_logger.info(
                f"The time delta at episode {episode_counter}, step {step_counter+1} is {delta} milliseconds which is {step_fps} fps."
            )

            episode.record_step(observations, rewards, dones, infos)
            step_counter += 1
        avg_frame_rate = fps_sum / (step_counter or 1)
        test_logger.info(
            f"Episode {episode_counter}, Minimum fps: {minimum_frame_rate}, Maximum fps: {maximum_frame_rate}, Average fps: {avg_frame_rate}."
        )
        assert (minimum_frame_rate >= 10) and (avg_frame_rate >= 20)
    env.close()
