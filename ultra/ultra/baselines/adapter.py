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
import numpy as np
import torch
from scipy.spatial import distance
import random, math, gym
from sys import path
from collections import OrderedDict
from ultra.baselines.common.baseline_state_preprocessor import BaselineStatePreprocessor
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml

path.append("./ultra")
from ultra.utils.common import (
    get_closest_waypoint,
    get_path_to_goal,
    ego_social_safety,
    rotate2d_vector,
)

seed = 0
random.seed(seed)
num_lookahead = 100


class BaselineAdapter:
    def __init__(self, agent_name):
        self.policy_params = load_yaml(
            f"ultra/baselines/{agent_name}/{agent_name}/params.yaml"
        )

        social_vehicle_params = self.policy_params["social_vehicles"]
        social_vehicle_params["observation_num_lookahead"] = self.policy_params[
            "observation_num_lookahead"
        ]
        self.observation_num_lookahead = social_vehicle_params[
            "observation_num_lookahead"
        ]
        self.num_social_features = social_vehicle_params["num_social_features"]
        self.social_capacity = social_vehicle_params["social_capacity"]
        self.social_vehicle_config = get_social_vehicle_configs(
            encoder_key=social_vehicle_params["encoder_key"],
            num_social_features=self.num_social_features,
            social_capacity=self.social_capacity,
            seed=social_vehicle_params["seed"],
            social_policy_hidden_units=social_vehicle_params[
                "social_policy_hidden_units"
            ],
            social_policy_init_std=social_vehicle_params["social_policy_init_std"],
        )

        self.social_vehicle_encoder = self.social_vehicle_config["encoder"]

        self.state_preprocessor = BaselineStatePreprocessor(
            social_vehicle_config=self.social_vehicle_config,
            observation_waypoints_lookahead=self.observation_num_lookahead,
            action_size=2,
        )

        self.social_feature_encoder_class = self.social_vehicle_encoder[
            "social_feature_encoder_class"
        ]
        self.social_feature_encoder_params = self.social_vehicle_encoder[
            "social_feature_encoder_params"
        ]

        self.state_size = self.state_preprocessor.num_low_dim_states
        if self.social_feature_encoder_class:
            self.state_size += self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            ).output_dim
        else:
            self.state_size += self.social_capacity * self.num_social_features

    @property
    def observation_space(self):
        low_dim_states_shape = self.state_preprocessor.num_low_dim_states
        if self.social_feature_encoder_class:
            social_vehicle_shape = self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            ).output_dim
        else:
            social_vehicle_shape = self.social_capacity * self.num_social_features
        return gym.spaces.Dict(
            {
                "low_dim_states": gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=(low_dim_states_shape,),
                    dtype=torch.Tensor,
                ),
                "social_vehicles": gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=(self.social_capacity, self.num_social_features),
                    dtype=torch.Tensor,
                ),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    # def action_adapter(self, model_action):
    #     # print why this doesn't go through?
    #     throttle, brake, steering = model_action
    #     # print(M)
    #     return np.array([throttle, brake, steering * np.pi * 0.25])

    def observation_adapter(self, env_observation):
        state = self.state_preprocessor(
            state=env_observation,
            observation_num_lookahead=self.observation_num_lookahead,
            social_capacity=self.social_capacity,
            social_vehicle_config=self.social_vehicle_config,
            # prev_action=self.prev_action
        )

        if len(state["social_vehicles"]) < self.social_capacity:
            remain = self.social_capacity - len(state["social_vehicles"])
            empty_social_vehicles = np.zeros(shape=(remain, 4), dtype=np.float32)
            state["social_vehicles"] = np.concatenate(
                (state["social_vehicles"], empty_social_vehicles)
            )
        # todo would this cause any issues for precog
        state["social_vehicles"] = state["social_vehicles"][: self.social_capacity]
        return state  # ego=ego, env_observation=env_observation)

    @staticmethod
    def reward_adapter(observation, reward):
        env_reward = reward
        ego_events = observation.events
        ego_observation = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )

        linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
        angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_observation.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.position)

        closest_wp, _ = get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=path,
            ego_position=ego_observation.position,
            ego_heading=ego_observation.heading,
        )
        angle_error = closest_wp.relative_heading(
            ego_observation.heading
        )  # relative heading radians [-pi, pi]

        # Distance from center
        signed_dist_from_center = closest_wp.signed_lateral_error(
            observation.ego_vehicle_state.position
        )
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
        ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
        ego_speed_reward = min(
            0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
        )  # m/s
        ego_collision = len(ego_events.collisions) > 0
        ego_collision_reward = -1.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
        ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        social_safety_reward = -0.02 if social_num_violations > 0 else 0
        ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
        ego_linear_jerk = -0.0001 * linear_jerk
        ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
        env_reward /= 100
        # DG: Different speed reward
        ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

        rewards = sum(
            [
                ego_goal_reward,
                ego_collision_reward,
                ego_off_road_reward,
                ego_off_route_reward,
                ego_wrong_way,
                ego_speed_reward,
                # ego_time_out,
                ego_dist_center_reward,
                ego_angle_error_reward,
                ego_reached_goal,
                ego_step_reward,
                env_reward,
                # ego_linear_jerk,
                # ego_angular_jerk,
                # ego_lat_speed,
                # ego_safety_reward,
                # social_safety_reward,
            ]
        )
        return rewards
