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
from itertools import cycle
import glob, yaml
from smarts.core.scenario import Scenario
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from ultra.baselines.adapter import BaselineAdapter
from ultra.baselines.common.yaml_loader import load_yaml
import numpy as np
from scipy.spatial import distance
import math, os
from sys import path

path.append("./ultra")
from ultra.utils.common import (
    get_closest_waypoint,
    get_path_to_goal,
    ego_social_safety,
)


class RLlibUltraEnv(RLlibHiWayEnv):
    def __init__(self, config):
        print(">>> ENV: ", config["eval_mode"])
        self.scenario_info = config["scenario_info"]
        self.scenarios = self.get_task(self.scenario_info[0], self.scenario_info[1])
        self._timestep_sec = config["timestep_sec"]
        self._seed = config["seed"]
        if not config["eval_mode"]:
            _scenarios = glob.glob(f"{self.scenarios['train']}")
        else:
            _scenarios = glob.glob(f"{self.scenarios['test']}")

        config["scenarios"] = _scenarios
        self.ultra_scores = BaselineAdapter.reward_adapter
        super().__init__(config=config)

        if config["ordered_scenarios"]:
            scenario_roots = []
            for root in config["scenarios"]:
                if Scenario.is_valid_scenario(root):
                    # The case that this is a scenario root
                    scenario_roots.append(root)
                else:
                    # The case that there this is a directory of scenarios: find each of the roots
                    scenario_roots.extend(Scenario.discover_scenarios(root))
            # Also see `smarts.env.HiwayEnv`
            self._scenarios_iterator = cycle(
                Scenario.variations_for_all_scenario_roots(
                    scenario_roots, list(config["agent_specs"].keys())
                )
            )

    def generate_logs(self, observation, highwayenv_score):
        ego_state = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )
        closest_wp, _ = get_closest_waypoint(
            num_lookahead=100,
            goal_path=path,
            ego_position=ego_state.position,
            ego_heading=ego_state.heading,
        )
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        linear_jerk = np.linalg.norm(ego_state.linear_jerk)
        angular_jerk = np.linalg.norm(ego_state.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_state.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.position)

        angle_error = closest_wp.relative_heading(
            ego_state.heading
        )  # relative heading radians [-pi, pi]

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        info = dict(
            position=ego_state.position,
            speed=ego_state.speed,
            steering=ego_state.steering,
            heading=ego_state.heading,
            dist_center=abs(ego_dist_center),
            start=start,
            goal=goal,
            closest_wp=closest_wp,
            events=observation.events,
            ego_num_violations=ego_num_violations,
            social_num_violations=social_num_violations,
            goal_dist=goal_dist,
            linear_jerk=np.linalg.norm(ego_state.linear_jerk),
            angular_jerk=np.linalg.norm(ego_state.angular_jerk),
            env_score=self.ultra_scores(observation, highwayenv_score),
        )
        return info

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self._agent_specs[agent_id].action_adapter(action)
            for agent_id, action in agent_actions.items()
        }

        observations, rewards, agent_dones, extras = self._smarts.step(agent_actions)

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

        infos = {
            agent_id: {"score": value, "env_obs": observations[agent_id]}
            for agent_id, value in extras["scores"].items()
        }

        # Ensure all contain the same agent_ids as keys
        assert (
            agent_actions.keys()
            == observations.keys()
            == rewards.keys()
            == infos.keys()
        )

        for agent_id in observations:
            agent_spec = self._agent_specs[agent_id]
            observation = observations[agent_id]
            reward = rewards[agent_id]
            info = infos[agent_id]

            rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
            observations[agent_id] = agent_spec.observation_adapter(observation)
            infos[agent_id] = agent_spec.info_adapter(observation, reward, info)
            infos[agent_id]["logs"] = self.generate_logs(observation, reward)

        for done in agent_dones.values():
            self._dones_registered += 1 if done else 0

        agent_dones["__all__"] = self._dones_registered == len(self._agent_specs)

        return observations, rewards, agent_dones, infos

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
            "timestep_sec": self._timestep_sec,
            "headless": self._headless,
        }

    @property
    def seed(self):
        return self._seed
