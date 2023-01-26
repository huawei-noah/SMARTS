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

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import smarts
from envision import types as envision_types
from envision.client import Client as Envision
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


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

            num_external_sumo_clients: Number of SUMO clients beyond SMARTS. Defaults to 0.

            seed: Random number generation seed. Defaults to 42.

            sumo_auto_start: True|False sumo will start automatically. Defaults to False.

            sumo_headless: True|False for sumo|sumo-gui. Defaults to False.

            sumo_port: Specifies sumo port. Defaults to None.

            fixed_timestep_sec: Step length for all components of the simulation. Defaults to 0.1 .
    """

    def __init__(self, config):
        self._log = logging.getLogger(self.__class__.__name__)
        seed = int(config.get("seed", 42))

        # See https://docs.ray.io/en/latest/rllib-env.html#configuring-environments
        # for context. We combine worker_index and vector_index through the Cantor pairing
        # function (https://en.wikipedia.org/wiki/Pairing_function) into a unique integer
        # and then add that to seed to both differentiate environment instances and
        # preserve determinism.
        a = config.worker_index
        b = config.vector_index
        c = (a + b) * (a + b + 1) // 2 + b
        smarts.core.seed(seed + c)

        self._agent_specs = config["agent_specs"]
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

        observations, rewards, dones, extras = self._smarts.step(agent_actions)

        # Agent termination: RLlib expects that we return a "last observation"
        # on the step that an agent transitions to "done". All subsequent calls
        # to env.step(..) will no longer contain actions from the "done" agent.
        #
        # The way we implement this behavior here is to rely on the presence of
        # agent actions to filter out all environment observations/rewards/infos
        # to only agents who are actively sending in actions.
        observations = {
            agent_id: obs
            for agent_id, obs in observations.items()
            if agent_id in agent_actions
        }
        rewards = {
            agent_id: reward
            for agent_id, reward in rewards.items()
            if agent_id in agent_actions
        }
        scores = {
            agent_id: score
            for agent_id, score in extras["scores"].items()
            if agent_id in agent_actions
        }

        infos = {key: {"score": value} for key, value in scores.items()}

        # Ensure all contain the same agent_ids as keys
        assert (
            agent_actions.keys()
            == observations.keys()
            == rewards.keys()
            == infos.keys()
        )
        for agent_id in agent_actions:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            observations[agent_id] = agent_spec.observation_adapter(observation)
            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

        for done in dones.values():
            self._dones_registered += 1 if done else 0
        dones["__all__"] = self._dones_registered >= len(self._agent_specs)

        return observations, rewards, dones, infos

    def reset(self):
        """Environment reset."""
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        if self._smarts is None:
            self._smarts = self._build_smarts()
            self._smarts.setup(scenario)

        env_observations = self._smarts.reset(scenario)

        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in env_observations.items()
        }

        return observations

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

        sumo_traffic = SumoTrafficSimulation(
            headless=self._sumo_headless,
            time_resolution=self._fixed_timestep_sec,
            num_external_sumo_clients=self._num_external_sumo_clients,
            sumo_port=self._sumo_port,
            auto_start=self._sumo_auto_start,
        )
        smarts_traffic = LocalTrafficProvider()

        sim = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=[sumo_traffic, smarts_traffic],
            envision=envision,
            fixed_timestep_sec=self._fixed_timestep_sec,
        )
        return sim
