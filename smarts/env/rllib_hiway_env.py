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
import warnings
from pathlib import Path
from typing import Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import smarts
from envision import etypes as envision_types
from envision.client import Client as Envision
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.env.utils.action_conversion import ActionOptions, ActionSpacesFormatter
from smarts.env.utils.observation_conversion import (
    ObservationOptions,
    ObservationSpacesFormatter,
)
from smarts.zoo.agent_spec import AgentSpec


class RLlibHiWayEnv(MultiAgentEnv):
    """This environment serves as a format to run multiple environments in
    parallel. This environment requires a specific configuration.

    Args:
        config (Dict[str,Any]): An environment configuration dictionary containing the following key value pairs.

            agent_specs: Dictionary mapping agent_ids to agent specs. Required.

            scenarios: List of scenario directories that will be run. Required.

            sim_name: A string to name this simulation. Defaults to None.

            envision_record_data_replay_path: Specifies Envision's data replay output directory. Defaults to None.

            envision_endpoint: Specifies Envision's uri. Defaults to None.

            headless: True|False envision disabled|enabled. Defaults to True.

            :spelling:ignore:`num_external_sumo_clients`: Number of SUMO clients beyond SMARTS. Defaults to 0.

            seed: Random number generation seed. Defaults to 42.

            sumo_auto_start: True|False sumo will start automatically. Defaults to False.

            sumo_headless: True|False for `sumo`|`sumo-gui`. Defaults to False.

            sumo_port: Specifies sumo port. Defaults to None.

            fixed_timestep_sec: Step length for all components of the simulation. Defaults to 0.1 .
    """

    def __init__(self, config):
        super().__init__()

        self._agent_specs: Dict[str, AgentSpec] = config["agent_specs"]
        agent_interfaces = {
            a_id: spec.interface for a_id, spec in self._agent_specs.items()
        }

        ## ---- Required environment attributes ----
        ## See ray/rllib/env/multi_agent_env.py
        self._agent_ids.update(id_ for id_ in self._agent_specs)

        action_options = ActionOptions.multi_agent
        self._action_formatter = ActionSpacesFormatter(
            agent_interfaces=agent_interfaces, action_options=action_options
        )
        self.action_space = self._action_formatter.space
        assert self.action_space is not None

        observation_options = ObservationOptions.multi_agent
        self._observations_formatter = ObservationSpacesFormatter(
            agent_interfaces=agent_interfaces, observation_options=observation_options
        )
        self.observation_space = self._observations_formatter.space
        assert self.observation_space is not None

        self._action_space_in_preferred_format = (
            self._check_if_action_space_maps_agent_id_to_sub_space()
        )
        self._obs_space_in_preferred_format = (
            self._check_if_obs_space_maps_agent_id_to_sub_space()
        )
        assert self._action_space_in_preferred_format is True
        ## ---- /Required environment attributes ----

        self._log = logging.getLogger(name=self.__class__.__name__)
        seed = int(config.get("seed", 42))

        # See https://docs.ray.io/en/latest/rllib-env.html#configuring-environments
        # for context. We combine worker_index and vector_index through the Cantor pairing
        # function (https://en.wikipedia.org/wiki/Pairing_function) into a unique integer
        # and then add that to seed to both differentiate environment instances and
        # preserve determinism.
        a = config.worker_index
        b = config.vector_index
        c = (a + b) * (a + b + 1) // 2 + b
        self._seed = seed + c
        smarts.core.seed(self._seed + c)

        self._scenarios = [
            str(Path(scenario).resolve()) for scenario in config["scenarios"]
        ]
        self._scenarios_iterator = Scenario.scenario_variations(
            self._scenarios,
            list(self._agent_specs.keys()),
        )

        self._sim_name = config.get("sim_name", None)
        self._headless = config.get("headless", True)
        self._num_external_sumo_clients = config.get("num_external_sumo_clients", 0)
        self._sumo_headless = config.get("sumo_headless", True)
        self._sumo_port = config.get("sumo_port")
        self._sumo_auto_start = config.get("sumo_auto_start", True)
        if "endless_traffic" in config:
            self._log.warning(
                "The endless_traffic option has been moved into Scenario Studio.  Please update your scenario code.",
            )

        self._envision_endpoint = config.get("envision_endpoint", None)
        self._envision_record_data_replay_path = config.get(
            "envision_record_data_replay_path", None
        )
        timestep_sec = config.get("timestep_sec")
        if timestep_sec:
            warnings.warn(
                "timestep_sec has been deprecated in favor of fixed_timestep_sec.  Please update your code.",
                category=DeprecationWarning,
            )
        self._fixed_timestep_sec = (
            config.get("fixed_timestep_sec") or timestep_sec or 0.1
        )
        self._smarts = None  # Created on env.setup()
        self._dones_registered = 0

    def step(self, agent_actions):
        """Environment step"""
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        assert isinstance(agent_actions, dict) and all(
            isinstance(key, str) for key in agent_actions.keys()
        ), "Expected Dict[str, any]"

        formatted_actions = self._action_formatter.format(agent_actions)
        env_observations, rewards, dones, extras = self._smarts.step(formatted_actions)
        env_observations = self._observations_formatter.format(env_observations)

        # Agent termination: RLlib expects that we return a "last observation"
        # on the step that an agent transitions to "done". All subsequent calls
        # to env.step(..) will no longer contain actions from the "done" agent.
        #
        # The way we implement this behavior here is to rely on the presence of
        # agent actions to filter out all environment observations/rewards/infos
        # to only agents who are actively sending in actions.
        observations = {
            agent_id: obs
            for agent_id, obs in env_observations.items()
            if agent_id in formatted_actions
        }
        rewards = {
            agent_id: reward
            for agent_id, reward in rewards.items()
            if agent_id in formatted_actions
        }
        scores = {
            agent_id: score
            for agent_id, score in extras["scores"].items()
            if agent_id in formatted_actions
        }

        infos = {
            agent_id: {
                "score": value,
                "reward": rewards[agent_id],
                "speed": observations[agent_id]["ego_vehicle_state"]["speed"],
            }
            for agent_id, value in scores.items()
        }

        # Ensure all contain the same agent_ids as keys
        assert (
            agent_actions.keys()
            == observations.keys()
            == rewards.keys()
            == infos.keys()
        )
        for agent_id in agent_actions:
            agent_spec = self._agent_specs[agent_id]
            observation = env_observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            observations[agent_id] = agent_spec.observation_adapter(observation)
            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

        for done in dones.values():
            self._dones_registered += 1 if done else 0
        dones["__all__"] = self._dones_registered >= len(self._agent_specs)

        return (
            observations,
            rewards,
            dones,
            dones,
            infos,
        )

    def reset(self, *, seed=None, options=None):
        """Environment reset."""
        if seed not in (None, 0):
            smarts.core.seed(self._seed + (seed or 0))

        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        if self._smarts is None:
            self._smarts = self._build_smarts()
            self._smarts.setup(scenario=scenario)

        env_observations = self._smarts.reset(scenario=scenario)

        env_observations = self._observations_formatter.format(
            observations=env_observations
        )
        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }
        info = {
            agent_id: {
                "score": 0,
                "reward": 0,
                "env_obs": agent_obs,
                "done": False,
                "map_source": self._smarts.scenario.road_map.source,
            }
            for agent_id, agent_obs in observations.items()
        }

        return observations, info

    def close(self):
        """Environment close."""
        if self._smarts is not None:
            self._smarts.destroy()

    def _build_smarts(self):
        agent_interfaces = {
            agent_id: spec.interface for agent_id, spec in self._agent_specs.items()
        }

        envision = None
        if not self._headless or self._envision_record_data_replay_path:
            envision = Envision(
                endpoint=self._envision_endpoint,
                sim_name=self._sim_name,
                output_dir=self._envision_record_data_replay_path,
                headless=self._headless,
            )
            preamble = envision_types.Preamble(scenarios=self._scenarios)
            envision.send(preamble)

        traffic_sims = []
        if Scenario.any_support_sumo_traffic(self._scenarios):
            from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

            sumo_traffic = SumoTrafficSimulation(
                headless=self._sumo_headless,
                time_resolution=self._fixed_timestep_sec,
                num_external_sumo_clients=self._num_external_sumo_clients,
                sumo_port=self._sumo_port,
                auto_start=self._sumo_auto_start,
            )
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        sim = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=traffic_sims,
            envision=envision,
            fixed_timestep_sec=self._fixed_timestep_sec,
        )
        return sim
