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
from collections import deque
from smarts.core.sensors import TopDownRGB
import unittest

import gym
import numpy as np
import ray

from smarts.core.agent_interface import (
    AgentInterface,
    NeighborhoodVehicles,
    RGB,
    Waypoints,
)
from smarts.core.agent import Agent, AgentSpec
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import make
from ultra.baselines.agent_spec import BaselineAgentSpec
from ultra.baselines.ppo.ppo.policy import PPOPolicy

AGENT_ID = "001"
timestep_sec = 0.1
seed = 2
task_id = "00"
task_level = "easy"


class EnvTest(unittest.TestCase):
    def test_scenario_task(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent()
            task_id1, task_level1 = env.scenario_info
            env.close()
            return task_id1, task_level1

        ray.init(ignore_reinit_error=True)
        task_id1, task_level1 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(task_id1 == task_id)
        self.assertTrue(task_level1 == task_level)

    def test_headless(self):
        @ray.remote(max_calls=1, num_gpus=0, num_cpus=1)
        def run_experiment():
            agent, env = prepare_test_env_agent(headless=False)
            headless1 = env.headless
            env.close()

            agent, env = prepare_test_env_agent(headless=True)
            headless2 = env.headless
            env.close()
            return headless1, headless2

        ray.init(ignore_reinit_error=True)
        headless1, headless2 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertFalse(headless1)
        self.assertTrue(headless2)

    def test_timestep_sec(self):
        @ray.remote(max_calls=1, num_gpus=0)
        def run_experiment():
            agent, env = prepare_test_env_agent(headless=False)
            timestep_sec1 = env.timestep_sec
            env.close()
            return timestep_sec1

        ray.init(ignore_reinit_error=True)
        timestep_sec1 = ray.get(run_experiment.remote())
        ray.shutdown()
        self.assertTrue(timestep_sec1 == timestep_sec)

    def test_observations_stacking(self):
        EPISODES = 3
        WIDTH = 64
        HEIGHT = WIDTH
        RESOLUTION = 50 / WIDTH
        ENVIRONMENT_STACK_SIZE = 4

        agent_spec = AgentSpec(
            interface=AgentInterface(
                waypoints=Waypoints(lookahead=1),
                neighborhood_vehicles=NeighborhoodVehicles(radius=10.0),
                rgb=RGB(width=WIDTH, height=HEIGHT, resolution=RESOLUTION),
                action=ActionSpaceType.Lane,
            ),
            agent_builder=TestLaneAgent,
        )
        agent = agent_spec.build_agent()

        environment = gym.make(
            "ultra.env:ultra-v0",
            agent_specs={AGENT_ID: agent_spec},
            scenario_info=("00", "easy"),
            headless=True,
            timestep_sec=0.1,
            seed=2,
        )

        def check_environment_observations_stack(environment):
            self.assertIsInstance(environment.smarts_observations_stack, deque)
            self.assertEqual(
                len(environment.smarts_observations_stack), ENVIRONMENT_STACK_SIZE
            )
            self.assertIsInstance(environment.smarts_observations_stack[0], dict)
            self.assertTrue(
                all(
                    str(environment.smarts_observations_stack[0]) == str(observations)
                    for observations in environment.smarts_observations_stack
                )
            )

        def check_stacked_observations(environment, observations):
            self.assertIn(AGENT_ID, observations)
            self.assertTrue(AGENT_ID, observations[AGENT_ID].top_down_rgb)
            self.assertIsInstance(observations[AGENT_ID].top_down_rgb, TopDownRGB)
            self.assertEqual(
                observations[AGENT_ID].top_down_rgb.metadata,
                environment.smarts_observations_stack[-1][
                    AGENT_ID
                ].top_down_rgb.metadata,
            )
            self.assertEqual(
                observations[AGENT_ID].top_down_rgb.data.shape,
                (ENVIRONMENT_STACK_SIZE, HEIGHT, WIDTH, 3),
            )
            # Ensure the stacked observation's TopDownRGB data is in the same order, and
            # and contains the same NumPy arrays as the environment's observation stack.
            self.assertTrue(
                all(
                    np.array_equal(
                        observations_from_stack[AGENT_ID].top_down_rgb.data,
                        observations[AGENT_ID].top_down_rgb.data[i],
                    )
                    for i, observations_from_stack in enumerate(
                        environment.smarts_observations_stack
                    )
                )
            )

        for _ in range(EPISODES):
            dones = {"__all__": False}
            observations = environment.reset()

            check_environment_observations_stack(environment)
            check_stacked_observations(environment, observations)

            while not dones["__all__"]:
                action = agent.act(observations[AGENT_ID])
                observations, _, dones, _ = environment.step({AGENT_ID: action})
                check_stacked_observations(environment, observations)

        environment.close()


# other attributes are not set
#'action_space', 'close', 'get_task', 'headless', 'info', 'metadata', 'observation_space', 'render', 'reset', 'reward_range', 'scenario_info', 'scenario_log', 'scenarios', 'seed', 'spec', 'step', 'timestep_sec', 'unwrapped'


class TestLaneAgent(Agent):
    def act(self, _):
        return "keep_lane"


def prepare_test_env_agent(headless=True):
    timestep_sec = 0.1
    # [throttle, brake, steering]
    policy_class = "ultra.baselines.ppo:ppo-v0"
    spec = make(locator=policy_class)
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )
    agent = spec.build_agent()
    return agent, env
