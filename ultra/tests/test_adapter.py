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
import random
from typing import Any, Callable, Dict, Tuple
import unittest

import gym
import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading
from smarts.core.events import Events
from smarts.core.scenario import PositionalGoal, Start
from smarts.core.sensors import Observation
from smarts.core.waypoints import Waypoint
import ultra.adapters as adapters
from ultra.env.ultra_env import UltraEnv

AGENT_ID = "001"
SEED = 2
TIMESTEP_SEC = 0.1


class AdapterTest(unittest.TestCase):
    def test_default_action_continuous_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultActionContinuous
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)
        space = adapters.space_from_type(ADAPTER_TYPE)

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            action_adapter=adapter,
        )
        action_sequence, _, _, _ = run_experiment(agent, environment)

        for action in action_sequence:
            self.assertIsInstance(action, np.ndarray)
            self.assertEqual(action.dtype, "float32")
            self.assertEqual(action.shape, (3,))
            self.assertGreaterEqual(action[0], 0.0)
            self.assertLessEqual(action[0], 1.0)
            self.assertGreaterEqual(action[1], 0.0)
            self.assertLessEqual(action[1], 1.0)
            self.assertGreaterEqual(action[2], -1.0)
            self.assertLessEqual(action[2], 1.0)
            self.assertEqual(space.dtype, action.dtype)
            self.assertEqual(space.shape, action.shape)
            self.assertTrue(space.contains(action))

    def test_default_action_discrete_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultActionDiscrete
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)
        space = adapters.space_from_type(ADAPTER_TYPE)

        AVAILABLE_ACTIONS = [
            "keep_lane",
            "slow_down",
            "change_lane_left",
            "change_lane_right",
        ]

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            action_adapter=adapter,
        )
        action_sequence, _, _, _ = run_experiment(agent, environment)

        for action in action_sequence:
            self.assertIsInstance(action, str)
            self.assertIn(action, AVAILABLE_ACTIONS)
            self.assertEqual(space.dtype, type(action))
            self.assertEqual(space.shape, ())
            self.assertTrue(space.contains(action))

    def test_default_info_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultInfo
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            info_adapter=adapter,
        )
        _, infos_sequence, _, _ = run_experiment(agent, environment, max_steps=1)

        infos = infos_sequence[0]
        self.assertIsInstance(infos, dict)
        self.assertIn(AGENT_ID, infos)
        self.assertIsInstance(infos[AGENT_ID], dict)
        self.assertIn("score", infos[AGENT_ID])
        self.assertIsInstance(infos[AGENT_ID]["score"], float)
        self.assertIn("env_obs", infos[AGENT_ID])
        self.assertIsInstance(infos[AGENT_ID]["env_obs"], Observation)
        self.assertIn("logs", infos[AGENT_ID])
        self.assertIsInstance(infos[AGENT_ID]["logs"], dict)
        self.assertIn("position", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["position"], np.ndarray)
        self.assertEqual(infos[AGENT_ID]["logs"]["position"].shape, (3,))
        self.assertIn("speed", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["speed"], float)
        self.assertIn("steering", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["steering"], float)
        self.assertIn("heading", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["heading"], Heading)
        self.assertIn("dist_center", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["dist_center"], float)
        self.assertIn("start", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["start"], Start)
        self.assertIn("goal", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["goal"], PositionalGoal)
        self.assertIn("closest_wp", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["closest_wp"], Waypoint)
        self.assertIn("events", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["events"], Events)
        self.assertIn("ego_num_violations", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["ego_num_violations"], int)
        self.assertIn("social_num_violations", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["social_num_violations"], int)
        self.assertIn("goal_dist", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["goal_dist"], float)
        self.assertIn("linear_jerk", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["linear_jerk"], float)
        self.assertIn("angular_jerk", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["angular_jerk"], float)
        self.assertIn("env_score", infos[AGENT_ID]["logs"])
        self.assertIsInstance(infos[AGENT_ID]["logs"]["env_score"], float)

    def test_default_observation_image_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultObservationImage
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)
        space = adapters.space_from_type(ADAPTER_TYPE)

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            observation_adapter=adapter,
        )
        _, _, observations_sequence, _ = run_experiment(agent, environment, max_steps=1)

        observations = observations_sequence[0]
        self.assertIsInstance(observations, dict)
        self.assertIn(AGENT_ID, observations)
        self.assertIsInstance(observations[AGENT_ID], np.ndarray)
        self.assertEqual(observations[AGENT_ID].dtype, "float32")
        self.assertEqual(observations[AGENT_ID].shape, (4, 64, 64))
        self.assertEqual(space.dtype, observations[AGENT_ID].dtype)
        self.assertEqual(space.shape, observations[AGENT_ID].shape)
        self.assertTrue(space.contains(observations[AGENT_ID]))

    def test_default_observation_vector_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultObservationVector
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)
        space = adapters.space_from_type(ADAPTER_TYPE)

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            observation_adapter=adapter,
        )
        _, _, observations_sequence, _ = run_experiment(agent, environment, max_steps=1)

        observations = observations_sequence[0]
        self.assertIsInstance(observations, dict)
        self.assertIn(AGENT_ID, observations)
        self.assertIn("low_dim_states", observations[AGENT_ID])
        self.assertIn("social_vehicles", observations[AGENT_ID])
        self.assertIsInstance(observations[AGENT_ID]["low_dim_states"], np.ndarray)
        self.assertIsInstance(observations[AGENT_ID]["social_vehicles"], np.ndarray)
        self.assertEqual(observations[AGENT_ID]["low_dim_states"].dtype, "float32")
        self.assertEqual(observations[AGENT_ID]["social_vehicles"].dtype, "float32")
        self.assertEqual(observations[AGENT_ID]["low_dim_states"].shape, (47,))
        self.assertEqual(observations[AGENT_ID]["social_vehicles"].shape, (10, 4))
        self.assertEqual(space.dtype, None)
        self.assertEqual(
            space["low_dim_states"].dtype,
            observations[AGENT_ID]["low_dim_states"].dtype,
        )
        self.assertEqual(
            space["social_vehicles"].dtype,
            observations[AGENT_ID]["social_vehicles"].dtype,
        )
        self.assertEqual(space.shape, None)
        self.assertEqual(
            space["low_dim_states"].shape,
            observations[AGENT_ID]["low_dim_states"].shape,
        )
        self.assertEqual(
            space["social_vehicles"].shape,
            observations[AGENT_ID]["social_vehicles"].shape,
        )
        self.assertTrue(space.contains(observations[AGENT_ID]))

    def test_default_reward_adapter(self):
        ADAPTER_TYPE = adapters.AdapterType.DefaultReward
        adapter = adapters.adapter_from_type(ADAPTER_TYPE)
        interface = adapters.required_interface_from_types(ADAPTER_TYPE)

        agent, environment = prepare_test_agent_and_environment(
            required_interface=interface,
            reward_adapter=adapter,
        )
        _, _, _, rewards_sequence = run_experiment(agent, environment, max_steps=1)

        rewards = rewards_sequence[0]
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(rewards[AGENT_ID], float)


class RandomAgent(Agent):
    def __init__(self, action_type: ActionSpaceType):
        if action_type == ActionSpaceType.Lane:
            # Actions in the form of strings.
            self._actions = [
                "keep_lane",
                "slow_down",
                "change_lane_left",
                "change_lane_right",
            ]
        elif action_type == ActionSpaceType.Continuous:
            # Actions in the form of np.array([throttle, brake, steering]), where
            # throttle and brake are in [0, 1] and steering is in [-1, 1].
            self._actions = np.random.uniform(low=-1.0, high=1.0, size=(50, 3))
            self._actions[:, :2] = (self._actions[:, :2] + 1) / 2
            self._actions = self._actions.astype(np.float32)
            self._actions = [action for action in self._actions]
        else:
            raise Exception(f"Unsupported ActionSpaceType: '{action_type}'.")

    def act(self, _: Observation):
        return random.choice(self._actions)


def prepare_test_agent_and_environment(
    required_interface: Dict[str, Any],
    action_adapter: Callable = lambda action: action,
    info_adapter: Callable = lambda observation, reward, info: info,
    observation_adapter: Callable = lambda observation: observation,
    reward_adapter: Callable = lambda _, reward: reward,
    headless: bool = True,
) -> Tuple[Agent, UltraEnv]:
    if "waypoints" not in required_interface:
        required_interface["waypoints"] = Waypoints(lookahead=20)
    if "neighborhood_vehicles" not in required_interface:
        required_interface["neighborhood_vehicles"] = NeighborhoodVehicles(radius=200)
    if "action" not in required_interface:
        required_interface["action"] = ActionSpaceType.Lane

    agent_spec = AgentSpec(
        interface=AgentInterface(**required_interface),
        agent_builder=RandomAgent,
        agent_params={"action_type": required_interface["action"]},
        action_adapter=action_adapter,
        info_adapter=info_adapter,
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
    infos_sequence = []
    observations_sequence = []
    rewards_sequence = []

    dones = {"__all__": False}

    observations = environment.reset()
    observations_sequence.append(observations)

    while not dones["__all__"] and len(action_sequence) <= max_steps:
        action = agent.act(observations[AGENT_ID])
        observations, rewards, dones, infos = environment.step({AGENT_ID: action})

        action_sequence.append(action)
        infos_sequence.append(infos)
        observations_sequence.append(observations)
        rewards_sequence.append(rewards)

    environment.close()

    return action_sequence, infos_sequence, observations_sequence, rewards_sequence
