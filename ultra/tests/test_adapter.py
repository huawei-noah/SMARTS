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
from typing import Tuple
import unittest

import gym
import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, Waypoints
from smarts.core.sensors import Observation
import ultra.adapters as adapters
from ultra.env.ultra_env import UltraEnv

AGENT_ID = "001"
SEED = 2
TIMESTEP_SEC = 0.1


class AdapterTest(unittest.TestCase):
    def test_default_action_continuous_adapter(self):
        agent, environment = prepare_test_agent_and_environment(
            action_type=adapters.AdapterType.DefaultActionContinuous,
            observation_type=adapters.AdapterType.DefaultObservationVector,
            reward_type=adapters.AdapterType.DefaultReward,
        )
        action_sequence, _, _ = run_experiment(agent, environment)

        for action in action_sequence:
            msg = f"Failed on action '{action}'."
            self.assertIsInstance(action, np.ndarray, msg=msg)
            self.assertEqual(action.dtype, "float32", msg=msg)
            self.assertEqual(action.shape, (3,), msg=msg)
            self.assertGreaterEqual(action[0], 0.0, msg=msg)
            self.assertLessEqual(action[0], 1.0, msg=msg)
            self.assertGreaterEqual(action[1], 0.0, msg=msg)
            self.assertLessEqual(action[1], 1.0, msg=msg)
            self.assertGreaterEqual(action[2], -1.0, msg=msg)
            self.assertLessEqual(action[2], 1.0, msg=msg)

    def test_default_action_discrete_adapter(self):
        AVAILABLE_ACTIONS = [
            "keep_lane",
            "slow_down",
            "change_lane_left",
            "change_lane_right",
        ]

        agent, environment = prepare_test_agent_and_environment(
            action_type=adapters.AdapterType.DefaultActionDiscrete,
            observation_type=adapters.AdapterType.DefaultObservationVector,
            reward_type=adapters.AdapterType.DefaultReward,
        )
        action_sequence, _, _ = run_experiment(agent, environment)

        for action in action_sequence:
            msg = f"Failed on action '{action}'."
            self.assertIsInstance(action, str, msg=msg)
            self.assertTrue(action in AVAILABLE_ACTIONS, msg=msg)

    def test_default_observation_image_adapter(self):
        agent, environment = prepare_test_agent_and_environment(
            action_type=adapters.AdapterType.DefaultActionDiscrete,
            observation_type=adapters.AdapterType.DefaultObservationImage,
            reward_type=adapters.AdapterType.DefaultReward,
        )
        _, observations_sequence, _ = run_experiment(agent, environment, max_steps=1)

        observations = observations_sequence[0]
        self.assertIsInstance(observations, dict)
        self.assertTrue(AGENT_ID in observations)
        self.assertIsInstance(observations[AGENT_ID], np.ndarray)
        self.assertEqual(observations[AGENT_ID].dtype, "float32")
        self.assertEqual(observations[AGENT_ID].shape, (1, 64, 64))

    def test_default_observation_vector_adapter(self):
        agent, environment = prepare_test_agent_and_environment(
            action_type=adapters.AdapterType.DefaultActionDiscrete,
            observation_type=adapters.AdapterType.DefaultObservationVector,
            reward_type=adapters.AdapterType.DefaultReward,
        )
        _, observations_sequence, _ = run_experiment(agent, environment, max_steps=1)

        observations = observations_sequence[0]
        self.assertIsInstance(observations, dict)
        self.assertTrue(AGENT_ID in observations)
        self.assertTrue("low_dim_states" in observations[AGENT_ID])
        self.assertTrue("social_vehicles" in observations[AGENT_ID])
        self.assertIsInstance(observations[AGENT_ID]["low_dim_states"], np.ndarray)
        self.assertIsInstance(observations[AGENT_ID]["social_vehicles"], np.ndarray)
        self.assertEqual(observations[AGENT_ID]["low_dim_states"].dtype, "float32")
        self.assertEqual(observations[AGENT_ID]["social_vehicles"].dtype, "float32")
        self.assertEqual(observations[AGENT_ID]["low_dim_states"].shape, (47,))
        self.assertEqual(observations[AGENT_ID]["social_vehicles"].shape, (10, 4))

    def test_default_reward_adapter(self):
        agent, environment = prepare_test_agent_and_environment(
            action_type=adapters.AdapterType.DefaultActionDiscrete,
            observation_type=adapters.AdapterType.DefaultObservationVector,
            reward_type=adapters.AdapterType.DefaultReward,
        )
        _, _, rewards_sequence = run_experiment(agent, environment, max_steps=1)

        rewards = rewards_sequence[0]
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(rewards[AGENT_ID], float)


class RandomAgent(Agent):
    def __init__(self, action_type: adapters.AdapterType):
        self._action_space = adapters.space_from_type(action_type)

    def act(self, _: Observation):
        return self._action_space.sample()


def prepare_test_agent_and_environment(
    action_type: adapters.AdapterType,
    observation_type: adapters.AdapterType,
    reward_type: adapters.AdapterType,
    headless=True,
) -> Tuple[Agent, UltraEnv]:
    action_adapter = adapters.adapter_from_type(action_type)
    observation_adapter = adapters.adapter_from_type(observation_type)
    reward_adapter = adapters.adapter_from_type(reward_type)
    required_interface = adapters.required_interface_from_types(
        action_type, observation_type, reward_type
    )

    if "waypoints" not in required_interface:
        required_interface["waypoints"] = Waypoints(lookahead=20)
    if "neighborhood_vehicles" not in required_interface:
        required_interface["neighborhood_vehicles"] = NeighborhoodVehicles(radius=200.0)

    agent_spec = AgentSpec(
        interface=AgentInterface(**required_interface),
        agent_builder=RandomAgent,
        agent_params={"action_type": action_type},
        action_adapter=action_adapter,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
    )
    agent = agent_spec.build_agent()

    environment = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: agent_spec},
        scenario_info=("00", "easy"),
        headless=headless,
        timestep_sec=TIMESTEP_SEC,
        seed=SEED,
    )

    return agent, environment


def run_experiment(agent: Agent, environment: UltraEnv, max_steps=30) -> Tuple:
    action_sequence = []
    observations_sequence = []
    rewards_sequence = []

    dones = {"__all__": False}

    observations = environment.reset()
    observations_sequence.append(observations)

    while not dones["__all__"] and len(action_sequence) <= max_steps:
        action = agent.act(observations[AGENT_ID])
        observations, reward, dones, _ = environment.step({AGENT_ID: action})

        action_sequence.append(action)
        observations_sequence.append(observations)
        rewards_sequence.append(reward)

    environment.close()

    return action_sequence, observations_sequence, rewards_sequence
