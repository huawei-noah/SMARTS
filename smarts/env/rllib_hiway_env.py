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
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np

import smarts
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


class RLlibHiWayEnv(MultiAgentEnv):
    """See MultiAgentEnv

    The purpose of this environment is as a format to run multiple environments in parallel.

    This environment will want a specific configuration:
        config: a dictionary with the environment configuration
            agent_specs:
                a dictionary of agent_ids to agents that run in the environment (required)
            scenarios:
                a list of directories of the scenarios that will be run (required)
            sim_name:
                a string that gives this simulation a name (default None)
            envision_record_data_replay_path:
                specify envision's data replay output directory (default None)
            envision_endpoint:
                used to specify envision's uri (default None)
            headless:
                true|false envision disabled (default True)
            num_external_sumo_clients:
                the number of SUMO clients beyond SMARTS (default 0)
            seed:
                the seed for random number generation (default 42)
            sumo_auto_start:
                true|false sumo will start automatically (default False)
            sumo_headless:
                true|false for sumo|sumo-gui (default False)
            sumo_port:
                used to specify a specific sumo port (default None)
            timestep_sec:
                the step length for all components of the simulation (default 0.1)
    """

    def __init__(self, config):
        seed = int(config.get("seed", 42))

        # See https://rllib.readthedocs.io/en/latest/rllib-env.html#configuring-environments
        # for context. We combine worker_index and vector_index through the Cantor pairing
        # function (https://en.wikipedia.org/wiki/Pairing_function) into a unique integer
        # and then add that to seed to both differentiate environment instances and
        # preserve determinism.
        a = config.worker_index
        b = config.vector_index
        c = (a + b) * (a + b + 1) // 2 + b
        smarts.core.seed(seed + c)

        self._agent_specs = config["agent_specs"]
        self._scenarios_iterator = Scenario.scenario_variations(
            config["scenarios"], list(self._agent_specs.keys()),
        )

        self._sim_name = config.get("sim_name", None)
        self._headless = config.get("headless", False)
        self._num_external_sumo_clients = config.get("num_external_sumo_clients", 0)
        self._sumo_headless = config.get("sumo_headless", True)
        self._sumo_port = config.get("sumo_port")
        self._sumo_auto_start = config.get("sumo_auto_start", True)
        self._endless_traffic = config.get("endless_traffic", True)

        self._envision_endpoint = config.get("envision_endpoint", None)
        self._envision_record_data_replay_path = config.get(
            "envision_record_data_replay_path", None
        )
        self._timestep_sec = config.get("timestep_sec", 0.1)
        self._smarts = None  # Created on env.setup()
        self._dones_registered = 0

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        observations, rewards, dones, extras = self._smarts.step(agent_actions)

        # Agent termination: RLlib expects that we return a "last observation"
        # on the step that an agent transitions to "done". All subsequent calls
        # to env.step(..) will no longer contain actions from the "done" agent.
        #
        # The way we implement this behaviour here is to rely on the presence of
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
        dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, dones, infos

    def reset(self):
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

        sim = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sim=SumoTrafficSimulation(
                headless=self._sumo_headless,
                time_resolution=self._timestep_sec,
                num_external_sumo_clients=self._num_external_sumo_clients,
                sumo_port=self._sumo_port,
                auto_start=self._sumo_auto_start,
                endless_traffic=self._endless_traffic,
            ),
            envision=envision,
            timestep_sec=self._timestep_sec,
        )
        return sim
