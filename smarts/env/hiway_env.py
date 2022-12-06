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
import os
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gym

from envision.client import Client as Envision
from envision.data_formatter import EnvisionDataFormatterArgs
from smarts.core import seed as smarts_seed
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.logging import timeit
from smarts.core.utils.visdom_client import VisdomClient
from smarts.zoo.agent_spec import AgentSpec


class HiWayEnv(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS."""

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_specs: Dict[str, AgentSpec],
        sim_name: Optional[str] = None,
        shuffle_scenarios: bool = True,
        headless: bool = True,
        visdom: bool = False,
        fixed_timestep_sec: Optional[float] = None,
        seed: int = 42,
        num_external_sumo_clients: int = 0,
        sumo_headless: bool = True,
        sumo_port: Optional[str] = None,
        sumo_auto_start: bool = True,
        envision_endpoint: Optional[str] = None,
        envision_record_data_replay_path: Optional[str] = None,
        zoo_addrs: Optional[str] = None,
        timestep_sec: Optional[
            float
        ] = None,  # for backwards compatibility (deprecated)
    ):
        """
        Args:
            scenarios (Sequence[str]):  A list of scenario directories that
                will be simulated.
            agent_specs (Dict[str, AgentSpec]): Specification of the agents
                that will run in the environment.
            sim_name (Optional[str], optional): Simulation name. Defaults to
                None.
            shuffle_scenarios (bool, optional): If true, order of scenarios
                will be randomized, else it will be maintained. Defaults to
                True.
            headless (bool, optional): If True, disables visualization in
                Envision. Defaults to False.
            visdom (bool, optional): If True, enables visualization of observed
                RGB images in Visdom. Defaults to False.
            fixed_timestep_sec (Optional[float], optional): Step duration for
                all components of the simulation. May be None if time deltas
                are externally-driven. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to 42.
            num_external_sumo_clients (int, optional): Number of SUMO clients
                beyond SMARTS. Defaults to 0.
            sumo_headless (bool, optional): If True, disables visualization in
                SUMO GUI. Defaults to True.
            sumo_port (Optional[str], optional): SUMO port. Defaults to None.
            sumo_auto_start (bool, optional): Automatic starting of SUMO.
                Defaults to True.
            envision_endpoint (Optional[str], optional): Envision's uri.
                Defaults to None.
            envision_record_data_replay_path (Optional[str], optional):
                Envision's data replay output directory. Defaults to None.
            zoo_addrs (Optional[str], optional): List of (ip, port) tuples of
                zoo server, used to instantiate remote social agents. Defaults
                to None.
            timestep_sec (Optional[float], optional): [description]. Defaults
                to None.
        """

        self._log = logging.getLogger(self.__class__.__name__)
        self.seed(seed)

        if timestep_sec and not fixed_timestep_sec:
            warnings.warn(
                "timestep_sec has been deprecated in favor of fixed_timestep_sec.  Please update your code.",
                category=DeprecationWarning,
            )
        if not fixed_timestep_sec:
            fixed_timestep_sec = timestep_sec or 0.1

        self._agent_specs = agent_specs
        self._dones_registered = 0

        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(agent_specs.keys()),
            shuffle_scenarios,
        )

        agent_interfaces = {
            agent_id: agent.interface for agent_id, agent in agent_specs.items()
        }

        envision_client = None
        if not headless or envision_record_data_replay_path:
            envision_client = Envision(
                endpoint=envision_endpoint,
                sim_name=sim_name,
                output_dir=envision_record_data_replay_path,
                headless=headless,
                data_formatter_args=EnvisionDataFormatterArgs(
                    "base", enable_reduction=False
                ),
            )

        self._env_renderer = None

        visdom_client = None
        if visdom:
            visdom_client = VisdomClient()

        traffic_sims = []
        if Scenario.any_support_sumo_traffic(scenarios):
            sumo_traffic = SumoTrafficSimulation(
                headless=sumo_headless,
                time_resolution=fixed_timestep_sec,
                num_external_sumo_clients=num_external_sumo_clients,
                sumo_port=sumo_port,
                auto_start=sumo_auto_start,
            )
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=traffic_sims,
            envision=envision_client,
            visdom=visdom_client,
            fixed_timestep_sec=fixed_timestep_sec,
            zoo_addrs=zoo_addrs,
        )

    @property
    def agent_specs(self) -> Dict[str, AgentSpec]:
        """Agents' specifications used in this simulation.

        Returns:
            (Dict[str, AgentSpec]): Agents' specifications.
        """
        return self._agent_specs

    @property
    def scenario_log(self) -> Dict[str, Union[float, str]]:
        """Simulation steps log.

        Returns:
            Dict[str, Union[float,str]]: A dictionary with the following keys.
                fixed_timestep_sec - Simulation timestep.
                scenario_map - Name of the current scenario.
                scenario_traffic - Traffic spec(s) used.
                mission_hash - Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_traffic": ",".join(map(os.path.basename, scenario.traffic_specs)),
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }


    @property
    def scenario(self):
        """Returns underlying scenario.

        Returns:
            Scenario: Current simulated scenario.
        """
        return self._smarts.scenario, self._smarts


    def seed(self, seed: int) -> int:
        """Sets random number generator seed number.

        Args:
            seed (int): Seed number.

        Returns:
            int: Seed number.
        """
        smarts_seed(seed)
        return seed

    def step(
        self, agent_actions
    ) -> Tuple[
        Dict[str, Observation], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        """Steps the environment.

        Args:
            agent_actions (Dict[str, Any]): Action taken for each agent.

        Returns:
            Tuple[ Dict[str, Observation], Dict[str, float], Dict[str, bool], Dict[str, Any] ]:
                Observations, rewards, dones, and infos for active agents.
        """
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        assert isinstance(agent_actions, dict) and all(
            isinstance(key, str) for key in agent_actions.keys()
        ), "Expected Dict[str, any]"

        observations, rewards, dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {"score": value, "env_obs": observations[agent_id]}
            for agent_id, value in extras["scores"].items()
        }

        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

        if self._env_renderer is not None:
            self._env_renderer.step(observations, rewards, dones, infos)

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_specs)

        return observations, rewards, dones, infos

    def reset(self) -> Dict[str, Observation]:
        """Reset the environment and initialize to the next scenario.

        Returns:
            Dict[str, Observation]: Agents' observation.
        """
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)

        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }
        if self._env_renderer is not None:
            self._env_renderer.reset(observations)

        return observations

    def render(self, mode="human"):
        """Renders according to metadata requirements."""

        if "rgb_array" in self.metadata["render.modes"]:
            if self._env_renderer is None:
                from smarts.env.utils.record import AgentCameraRGBRender

                self._env_renderer = AgentCameraRGBRender(self)

            return self._env_renderer.render(env=self)

    def close(self):
        """Closes the environment and releases all resources."""
        if self._smarts is not None:
            self._smarts.destroy()