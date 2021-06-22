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
from collections import deque
import copy
import glob
import math
import os
from itertools import cycle
from sys import path
from typing import Dict

import numpy as np
import yaml, inspect
from scipy.spatial import distance

from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation, TopDownRGB
from smarts.env.hiway_env import HiWayEnv
import ultra.adapters as adapters
from ultra.baselines.common.yaml_loader import load_yaml

path.append("./ultra")


_STACK_SIZE = 4


class UltraEnv(HiWayEnv):
    def __init__(
        self,
        agent_specs,
        scenario_info,
        headless,
        timestep_sec,
        seed,
        eval_mode=False,
        ordered_scenarios=False,
    ):
        self.timestep_sec = timestep_sec
        self.headless = headless
        self.scenario_info = scenario_info
        self.scenarios = self.get_task(scenario_info[0], scenario_info[1])
        if not eval_mode:
            _scenarios = glob.glob(f"{self.scenarios['train']}")
        else:
            _scenarios = glob.glob(f"{self.scenarios['test']}")

        self.smarts_observations_stack = deque(maxlen=_STACK_SIZE)

        super().__init__(
            scenarios=_scenarios,
            agent_specs=agent_specs,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
            visdom=False,
        )

        if ordered_scenarios:
            scenario_roots = []
            for root in _scenarios:
                if Scenario.is_valid_scenario(root):
                    # The case that this is a scenario root
                    scenario_roots.append(root)
                else:
                    # The case that there this is a directory of scenarios: find each of the roots
                    scenario_roots.extend(Scenario.discover_scenarios(root))
            # Also see `smarts.env.HiwayEnv`
            self._scenarios_iterator = cycle(
                Scenario.variations_for_all_scenario_roots(
                    scenario_roots, list(agent_specs.keys())
                )
            )

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        smarts_observations, rewards, agent_dones, extras = self._smarts.step(
            agent_actions
        )

        self.smarts_observations_stack.append(copy.deepcopy(smarts_observations))
        observations = self._adapt_smarts_observations(smarts_observations)

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

        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, agent_dones, infos

    def reset(self):
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        smarts_observations = self._smarts.reset(scenario)

        for _ in range(_STACK_SIZE):
            self.smarts_observations_stack.append(copy.deepcopy(smarts_observations))
        observations = self._adapt_smarts_observations(smarts_observations)

        observations = {
            agent_id: self._agent_specs[agent_id].observation_adapter(obs)
            for agent_id, obs in observations.items()
        }

        return observations

    def get_task(self, task_id, task_level):
        base_dir = os.path.join(os.path.dirname(__file__), "../")
        config_path = os.path.join(base_dir, "config.yaml")

        with open(config_path, "r") as task_file:
            scenarios = yaml.safe_load(task_file)["tasks"]
            task = scenarios[f"task{task_id}"][task_level]

        task["train"] = os.path.join(base_dir, task["train"])
        task["test"] = os.path.join(base_dir, task["test"])
        return task

    @property
    def info(self):
        return {
            "scenario_info": self.scenario_info,
            "timestep_sec": self.timestep_sec,
            "headless": self.headless,
        }

    def _adapt_smarts_observations(
        self, current_observations: Dict[str, Observation]
    ) -> Dict[str, Observation]:
        """Adapts the observations received from the SMARTS simulator.

        The ULTRA environment slightly adapts the simulator observations by:
        - Stacking the TopDownRGB component's data of each observation if the TopDownRGB
          component of the observation is not None.

        Args:
            current_observations (Dict[str, Observation]): The current simulator
                observations.

        Returns:
            Dict[str, Observation]: The adapted current observations.
        """
        for agent_id, current_observation in current_observations.items():
            if current_observation.top_down_rgb:
                # This agent's observation contains a TopDownRGB, stack its data.
                current_top_down_rgb = current_observation.top_down_rgb

                top_down_rgb_data = []
                for observations in self.smarts_observations_stack:
                    if agent_id in observations:
                        top_down_rgb_data.append(
                            observations[agent_id].top_down_rgb.data
                        )
                    else:
                        # Use the current observation's TopDownRGB data if this agent
                        # doesn't have previous observations to use to build the stack.
                        top_down_rgb_data.append(current_top_down_rgb.data)
                stacked_top_down_rgb_data = np.stack(top_down_rgb_data)

                # Create the new TopDownRGB with stacked data.
                stacked_top_down_rgb = TopDownRGB(
                    metadata=current_top_down_rgb.metadata,
                    data=stacked_top_down_rgb_data,
                )

                current_observations[agent_id].top_down_rgb = stacked_top_down_rgb

        return current_observations
