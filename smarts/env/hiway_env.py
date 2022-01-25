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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import os
import warnings
from typing import Dict, Sequence

import gym

from envision.client import Client as Envision
from smarts.core import seed as smarts_seed
from smarts.core.agent import AgentSpec
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.utils.logging import timeit
from smarts.core.utils.visdom_client import VisdomClient


class HiWayEnv(gym.Env):
    """A complete gym environment that wraps a SMARTS simulation.

    Args:
        scenarios:
            a list of directories of the scenarios that will be run
        agent_specs:
            a list of agents that will run in the environment
        sim_name:
            a string that gives this simulation a name
        headless:
            true|false envision disabled
        visdom:
            true|false visdom integration
        fixed_timestep_sec:
            the step length for all components of the simulation
            (may be None if time deltas are externally-driven)
        seed:
            the seed for random number generation
        num_external_sumo_clients:
            the number of SUMO clients beyond SMARTS
        sumo_headless:
            true|false for SUMO visualization disabled [sumo-gui|sumo]
        sumo_port:
            used to specify a specific sumo port
        sumo_auto_start:
            true|false sumo will start automatically
        envision_endpoint:
            used to specify envision's uri
        envision_record_data_replay_path:
            used to specify envision's data replay output directory
        zoo_addrs:
            List of (ip, port) tuples of zoo server, used to instantiate remote social agents
    """

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_specs: Dict[str, AgentSpec],
        sim_name=None,
        shuffle_scenarios=True,
        headless=False,
        visdom=False,
        fixed_timestep_sec=None,
        seed=42,
        num_external_sumo_clients=0,
        sumo_headless=True,
        sumo_port=None,
        sumo_auto_start=True,
        endless_traffic=True,
        envision_endpoint=None,
        envision_record_data_replay_path=None,
        zoo_addrs=None,
        timestep_sec=None,  # for backwards compatibility (deprecated)
    ):
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
            )

        visdom_client = None
        if visdom:
            visdom_client = VisdomClient()

        all_sumo = Scenario.supports_traffic_simulation(scenarios)
        traffic_sim = None
        if not all_sumo:
            # We currently only support the Native SUMO Traffic Provider and Social Agents for SUMO maps
            if zoo_addrs:
                warnings.warn("`zoo_addrs` can only be used with SUMO scenarios")
                zoo_addrs = None
            warnings.warn(
                "We currently only support the Native SUMO Traffic Provider and Social Agents for SUMO maps."
                "All scenarios passed need to be of SUMO, to enable SUMO Traffic Simulation and Social Agents."
            )
            pass
        else:
            from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

            traffic_sim = SumoTrafficSimulation(
                headless=sumo_headless,
                time_resolution=fixed_timestep_sec,
                num_external_sumo_clients=num_external_sumo_clients,
                sumo_port=sumo_port,
                auto_start=sumo_auto_start,
                endless_traffic=endless_traffic,
            )
            zoo_addrs = zoo_addrs

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sim=traffic_sim,
            envision=envision_client,
            visdom=visdom_client,
            fixed_timestep_sec=fixed_timestep_sec,
            zoo_addrs=zoo_addrs,
        )

    @property
    def agent_specs(self):
        """Agent specs currently in use for this simulation.

        Returns:
            A list of AgentSpec.
        """
        return self._agent_specs

    @property
    def scenario_log(self):
        """Simulation step logs.

        Returns:
            A dictionary with the following:
                fixed_timestep_sec:
                    The timestep of the simulation.
                scenario_map:
                    The name of the current scenario.
                scenario_routes:
                    The routes in the map.
                mission_hash:
                    The hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_routes": scenario.route or "",
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    def seed(self, seed: int) -> int:
        """Set the seed of this environment."""
        smarts_seed(seed)
        return seed

    def step(self, agent_actions):
        """Step and return observations, rewards, dones, and infos."""
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        observations, rewards, dones, extras = None, None, None, None
        with timeit("SMARTS Simulation/Scenario Step", self._log):
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

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_specs)

        return observations, rewards, dones, infos

    def reset(self):
        """Reset the environment and reinitialize to the next scenario."""
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        env_observations = self._smarts.reset(scenario)

        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }

        return observations

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        """Clean up all remaining resources."""
        if self._smarts is not None:
            self._smarts.destroy()
            self._smarts = None
