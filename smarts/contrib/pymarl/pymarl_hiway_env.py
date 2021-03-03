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
import numpy as np
from gym.spaces import Box, Discrete

import smarts
from envision.client import Client as Envision
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

from .adapters.action_adapter import (
    DEFAULT_ACTION_SPACE,
    N_ACTIONS,
    default_action_adapter,
)
from .adapters.observation_adapter import DEFAULT_OBSERVATION_SPACE, default_obs_adapter
from .adapters.reward_adapter import default_reward_adapter
from .adapters.state_adapter import DEFAULT_STATE_SPACE, default_state_adapter


class PyMARLHiWayEnv:
    """This class adheres to the PyMARL MultiAgentEnv so it can be run by PyMARL.
    See: https://git.io/JvMb9

    This environment will want a specific configuration:
      config: a dictionary with the environment configuration
        agent_specs:
            a dictionary of agent_ids to agents that will run in the environment (required)
        scenarios:
            a list of directories of the scenarios that will be run (required)
        sim_name:
            a string that gives this simulation a name (default None)
        envision_record_data_replay_path:
            used to specify envision's data replay output directory (default None)
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
        self._config = config

        # XXX: These are intentionally left public at PyMARL's request
        self.n_agents = config.get("n_agents", 1)
        self.episode_limit = config.get("episode_limit", 1000)
        self.observation_space = config.get(
            "observation_space", DEFAULT_OBSERVATION_SPACE
        )
        self.action_space = config.get("action_space", DEFAULT_ACTION_SPACE)
        self.state_space = config.get("state_space", DEFAULT_STATE_SPACE)

        self._agent_ids = ["Agent %i" % i for i in range(self.n_agents)]

        self._reward_adapter = config.get("reward_adapter", default_reward_adapter)
        self._observation_adapter = config.get(
            "observation_adapter", default_obs_adapter
        )
        self._action_adapter = config.get("action_adapter", default_action_adapter)
        self._done_adapter = config.get(
            "done_adapter", lambda dones: list(dones.values())
        )
        self._state_adapter = config.get("state_adapter", default_state_adapter)

        self._headless = config.get("headless", False)
        self._timestep_sec = config.get("timestep_sec", 0.01)
        self._observations = None
        self._state = None
        self._steps = 0

        seed = self._config.get("seed", 42)
        smarts.core.seed(seed)

        self._scenarios_iterator = Scenario.scenario_variations(
            config["scenarios"], self._agent_ids
        )

        agent_interfaces = {
            agent_id: AgentInterface.from_type(
                config.get("agent_type", AgentType.Laner),
                max_episode_steps=self.episode_limit,
                debug=config.get("debug", False),
            )
            for i, agent_id, in enumerate(self._agent_ids)
        }

        envision = None
        if not self._headless or config.get("envision_record_data_replay_path", None):
            envision = Envision(
                endpoint=config.get("envision_endpoint", None),
                sim_name=config.get("sim_name", None),
                output_dir=config.get("envision_record_data_replay_path", None),
            )

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sim=SumoTrafficSimulation(time_resolution=self._timestep_sec),
            envision=envision,
            timestep_sec=self._timestep_sec,
        )

    def get_obs(self):
        return self._observations

    def get_obs_agent(self, agent_id):
        return self._observations[agent_id]

    def get_obs_size(self):
        obs_size = 0
        for obs in self.observation_space.spaces.values():
            if type(obs) is Box:
                obs_size += np.prod(obs.shape)
            elif type(obs) is Discrete:
                obs_size += obs.n
        return obs_size

    def get_state(self):
        return np.concatenate(self._observations)

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [np.ones((N_ACTIONS,)) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return np.ones((N_ACTIONS,))

    def get_total_actions(self):
        return N_ACTIONS

    def render(self):
        pass

    def save_replay(self):
        pass

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self._action_adapter(action)
            for agent_id, action in zip(self._agent_ids, agent_actions)
        }

        observations, rewards, dones, extras = self._smarts.step(agent_actions)
        infos = {
            f"score_{i}": score for i, score in enumerate(extras["scores"].values())
        }

        # Ensure observations contain the same keys as rewards
        assert observations.keys() == rewards.keys()
        self._observations = np.asarray(
            [
                np.concatenate(list(self._observation_adapter(obs).values()))
                for obs in observations.values()
            ]
        )
        rewards = [
            self._reward_adapter(obs, rew)
            for obs, rew in zip(observations.values(), rewards.values())
        ]

        infos["rewards_list"] = rewards

        self._steps += 1
        infos["dones_list"] = np.array(list(dones.values()))
        dones = infos["dones_list"]
        if self._steps >= self.episode_limit:
            infos["episode_steps"] = self._steps
            dones = np.array([True])

        return np.mean(rewards), dones, infos

    def reset(self):
        self._steps = 0

        scenario = next(self._scenarios_iterator)
        observations = self._smarts.reset(scenario)
        self._observations = np.asarray(
            [
                np.concatenate(list(self._observation_adapter(obs).values()))
                for obs in observations.values()
            ]
        )
        return self._observations

    def close(self):
        if self._smarts is not None:
            self._smarts.destroy()

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
