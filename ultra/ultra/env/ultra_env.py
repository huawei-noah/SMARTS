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

import copy
import glob
import numpy as np
import pathlib
import yaml

from collections import deque
from itertools import cycle
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation, TopDownRGB
from smarts.env.hiway_env import HiWayEnv
from sys import path
from typing import Dict, List, Tuple, Sequence, Union

path.append("./ultra")

_CONFIG_FILE = "config.yaml"

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
        self.scenario_info = scenario_info
        self.headless = headless
        self.timestep_sec = timestep_sec
        self.smarts_observations_stack = deque(maxlen=_STACK_SIZE)

        scenarios = UltraEnv.get_scenarios_from_scenario_info(scenario_info, eval_mode)

        super().__init__(
            scenarios=scenarios,
            agent_specs=agent_specs,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
            visdom=False,
            endless_traffic=False,
        )

        if ordered_scenarios:
            scenario_roots = []
            for root in scenarios:
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

    @staticmethod
    def get_scenarios_from_scenario_info(
        scenario_info: Union[Tuple[str, str], Sequence[str]], eval_mode: bool = False
    ) -> List[str]:
        """Finds all scenarios from a given (task, level) tuple, or sequence of scenario
        directories.

        Args:
            scenario_info (Union[Tuple[str, str], Sequence[str]]): Either a tuple of
                two strings (task, level) that describe the scenarios of the specific
                task and level, or a sequence of scenario directory strings.
            eval_mode (bool): Used only when obtaining scenarios from a (task, level)
                tuple. This determines whether to return the evaluation scenarios of the
                specified task's level. Training scenarios are returned by default.

        Returns:
            List[str]: A list of scenario directory strings.
        """
        try:
            # Attempt to load scenarios from a (task, level) tuple.
            task, level = scenario_info
            base_dir = pathlib.Path(__file__).parent.parent
            task_config_path = base_dir / _CONFIG_FILE

            with open(task_config_path, "r") as tasks_file:
                tasks = yaml.safe_load(tasks_file)["tasks"]
            scenario_paths = tasks[f"task{task}"][level]

            scenario_paths["train"] = (base_dir / scenario_paths["train"]).resolve()
            scenario_paths["test"] = (base_dir / scenario_paths["test"]).resolve()

            if not eval_mode:
                scenarios = glob.glob(f"{scenario_paths['train']}")
            else:
                scenarios = glob.glob(f"{scenario_paths['test']}")
        except (KeyError, ValueError):
            # Treat scenario_info as a list of scenario directories.
            scenarios = scenario_info
        except Exception as exception:
            # Otherwise, something else has gone wrong.
            raise exception

        return scenarios

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
