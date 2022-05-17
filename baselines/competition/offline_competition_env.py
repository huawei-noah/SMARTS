# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import gym.spaces as spaces
import numpy as np
from envision.client import Client as Envision
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.utils.logging import timeit
from smarts.core.utils.math import vec_to_radians

NUM_WAYPOINTS = 5
NUM_NEIGHBORS = 10
MAX_MPS = 100
AGENT_ID = "EGO"


def _filter(obs: Observation, env):
    def _clip(formatted_obs, observation_space):
        result = {}
        for k, v in formatted_obs.items():
            if type(observation_space[k]) == spaces.Tuple:
                tuple_vals = []
                for inner_val, inner_space in zip(v, observation_space[k]):
                    tuple_vals.append(
                        np.clip(inner_val, inner_space.low, inner_space.high)
                    )
                result[k] = tuple(tuple_vals)
            else:
                result[k] = np.clip(
                    v, observation_space[k].low, observation_space[k].high
                )
        return result

    neighbors = [np.array([np.inf] * 5)] * NUM_NEIGHBORS
    if obs.neighborhood_vehicle_states:
        for i in range(min(len(obs.neighborhood_vehicle_states), NUM_NEIGHBORS)):
            neighbors[i] = np.hstack(
                (
                    obs.neighborhood_vehicle_states[i].position,
                    np.array(
                        [
                            float(obs.neighborhood_vehicle_states[i].heading),
                            obs.neighborhood_vehicle_states[i].speed,
                        ]
                    ),
                )
            )

    obs = {
        "position": obs.ego_vehicle_state.position,
        "linear_velocity": obs.ego_vehicle_state.linear_velocity,
        "neighbors": tuple(neighbors),
    }
    obs = _clip(obs, env.observation_space)
    assert env.observation_space.contains(
        obs
    ), "Observation mismatch with observation space. Less keys in observation space dictionary."
    return obs


class OfflineCompetitionEnv(gym.Env):
    """A specific competition environment."""

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""
    action_space = spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float)
    observation_space = spaces.Dict(
        {
            # position x, y, z in meters
            "position": spaces.Box(
                low=-math.inf,
                high=math.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            # Velocity
            "linear_velocity": spaces.Box(
                low=-MAX_MPS,
                high=MAX_MPS,
                shape=(3,),
                dtype=np.float32,
            ),
            # neighboring vehicle states (3D position, heading, speed)
            "neighbors": spaces.Tuple(
                tuple(
                    [
                        spaces.Box(
                            low=-math.inf, high=math.inf, shape=(5,), dtype=np.float32
                        )
                    ]
                    * NUM_NEIGHBORS
                )
            ),
        }
    )

    def __init__(
        self,
        scenarios: Sequence[str],
        headless: bool = True,
        sim_name: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        seed: int = 42,
        envision_endpoint: Optional[str] = None,
        envision_record_data_replay_path: Optional[str] = None,
    ):
        """
        Args:
            sim_name (Optional[str], optional): Simulation name. Defaults to
                None.
            headless (bool, optional): If True, disables visualization in
                Envision. Defaults to True.
            seed (int, optional): Random number generator seed. Defaults to 42.
            envision_endpoint (Optional[str], optional): Envision's uri.
                Defaults to None.
            envision_record_data_replay_path (Optional[str], optional):
                Envision's data replay output directory. Defaults to None.
            recorded_obs_path (Optional[Path, str], optional):
                Output directory to write recorded observations to as a csv file.
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self.seed(seed)
        self._current_time = 0.0
        self._fixed_timestep_sec = 0.1

        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            [AGENT_ID],
        )

        agent_interface = AgentInterface(
            max_episode_steps=max_episode_steps,
            drivable_area_grid_map=True,
            waypoints=True,
            neighborhood_vehicles=True,
            action=ActionSpaceType.TargetPose,
        )

        envision_client = None
        if not headless or envision_record_data_replay_path:
            envision_client = Envision(
                endpoint=envision_endpoint,
                sim_name=sim_name,
                output_dir=envision_record_data_replay_path,
                headless=headless,
            )

        from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

        traffic_sim = SumoTrafficSimulation(
            headless=True,
            time_resolution=self._fixed_timestep_sec,
            endless_traffic=False,
        )

        self._smarts = SMARTS(
            agent_interfaces={AGENT_ID: agent_interface},
            traffic_sim=traffic_sim,
            envision=envision_client,
            fixed_timestep_sec=self._fixed_timestep_sec,
        )

        self._last_obs = None
        self._last_veh_obs = None

    def seed(self, seed: int) -> List[int]:
        """Sets random number generator seed number.

        Args:
            seed (int): Seed number.

        Returns:
            list[int]: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'.
        """
        assert isinstance(seed, int), "Seed value must be an integer."
        smarts_seed(seed)
        return [seed]

    def step(
        self, agent_action: Tuple[float, float]
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Steps the environment.

        Args:
            agent_action (Tuple[float, float]): Action taken by the agent.

        Returns:
            Tuple[Observation, float, bool, Any]:
                Observation, reward, done, and info for the environment.
        """
        assert self.action_space.contains(
            np.array(agent_action)
        ), f"Action {agent_action} must be within the action space: {self.action_space}"
        observation, reward, done, extra = None, None, None, None

        heading = Heading(vec_to_radians(agent_action))  # naive heading
        l_p = self._last_obs.ego_vehicle_state.position
        target_pose = np.array(
            [
                l_p[0] + agent_action[0],
                l_p[1] + agent_action[1],
                float(heading),
                self._fixed_timestep_sec,
            ]
        )

        with timeit("SMARTS Simulation/Scenario Step", self._log):
            observations, rewards, dones, extras = self._smarts.step(
                {AGENT_ID: target_pose}
            )

        done = dones[AGENT_ID]
        reward = rewards[AGENT_ID]
        extra = {"score": extras["scores"][AGENT_ID], "env_obs": observations[AGENT_ID]}
        observation = observations[AGENT_ID]

        self._last_obs = observation
        self._current_time += observation.dt

        return (
            _filter(observation, self),
            reward,
            done,
            extra,
        )

    def reset(self) -> Observation:
        """Reset the environment and initialize to the next scenario.

        Returns:
            Observation: Agents' observation.
        """
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)

        observation = env_observations[AGENT_ID]
        self._last_obs = observation

        self._current_time += observation.dt

        return _filter(observation, self)

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        """Closes the environment and releases all resources."""
        if self._smarts is not None:
            self._smarts.destroy()
            self._smarts = None
