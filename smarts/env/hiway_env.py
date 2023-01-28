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
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import gym

from envision import types as envision_types
from envision.client import Client as Envision
from envision.data_formatter import EnvisionDataFormatterArgs
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.observations import Observation
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.visdom_client import VisdomClient
from smarts.zoo.agent_spec import AgentSpec


class HiWayEnv(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS.

    Args:
        scenarios (Sequence[str]):  A list of scenario directories that
            will be simulated.
        agent_specs (Dict[str, AgentSpec]): Specification of the agents
            that will run in the environment. (Deprecated in favor of
            agent_interfaces)
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
        agent_interfaces (Dict[str, AgentInterface]): Specification of the agents
            needs that will be used to configure the environment.
    """

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_specs: Optional[Dict[str, AgentSpec]] = None,  # (deprecated)
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
        agent_interfaces: Optional[Dict[str, AgentInterface]] = None,
        timestep_sec: Optional[
            float
        ] = None,  # for backwards compatibility (deprecated)
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self.seed(seed)

        if timestep_sec and not fixed_timestep_sec:
            warnings.warn(
                "timestep_sec has been deprecated in favor of fixed_timestep_sec.  Please update your code.",
                category=DeprecationWarning,
            )
        if agent_specs is not None:
            warnings.warn(
                "agent_specs has been deprecated in favor of agent_interfaces.  Please update your code.",
                category=DeprecationWarning,
            )
        if not fixed_timestep_sec:
            fixed_timestep_sec = timestep_sec or 0.1

        self._agent_interfaces = agent_interfaces or {}
        if isinstance(agent_interfaces, dict):
            self._agent_specs = {
                a_id: AgentSpec(a_inter) for a_id, a_inter in agent_interfaces.items()
            }
        elif isinstance(agent_specs, dict):
            self._agent_specs = agent_specs
            self._agent_interfaces = {
                agent_id: agent.interface for agent_id, agent in agent_specs.items()
            }
        else:
            raise TypeError(
                f"agent_interface must be supplied as `{Dict[str, AgentInterface]}`."
            )
        self._dones_registered = 0

        scenarios = [str(Path(scenario).resolve()) for scenario in scenarios]
        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(self._agent_interfaces.keys()),
            shuffle_scenarios,
        )

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
            preamble = envision_types.Preamble(scenarios=scenarios)
            envision_client.send(preamble)

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
            agent_interfaces=self._agent_interfaces,
            traffic_sims=traffic_sims,
            envision=envision_client,
            visdom=visdom_client,
            fixed_timestep_sec=fixed_timestep_sec,
            zoo_addrs=zoo_addrs,
        )

    @property
    def agent_ids(self) -> Set[str]:
        """Agent ids of all agents that potentially will be in the environment.
        Returns:
            (Set[str]): Agent ids.
        """
        return set(self._agent_interfaces)

    @property
    def agent_interfaces(self) -> Dict[str, AgentInterface]:
        """Agents' interfaces used in this simulation.

        Returns:
            (Dict[str, AgentInterface]): Agents' interfaces.
        """
        return self._agent_interfaces

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

                + fixed_timestep_sec: Simulation timestep.
                + scenario_map: Name of the current scenario.
                + scenario_traffic: Traffic spec(s) used.
                + mission_hash: Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_traffic": ",".join(map(os.path.basename, scenario.traffic_specs)),
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    @property
    def scenario(self) -> Scenario:
        """Returns underlying scenario.

        Returns:
            Current simulated scenario.
        """
        return self._smarts.scenario

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
        assert isinstance(agent_actions, dict) and all(
            isinstance(key, str) for key in agent_actions.keys()
        ), "Expected Dict[str, any]"

        observations, rewards, dones, extras = self._smarts.step(agent_actions)

        infos = {
            agent_id: {
                "score": value,
                "env_obs": observations[agent_id],
                "done": dones[agent_id],
            }
            for agent_id, value in extras["scores"].items()
        }

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
        observations = self._smarts.reset(scenario)

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
