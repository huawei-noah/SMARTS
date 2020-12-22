import gym
from smarts.env.custom_observations import Adapter
import numpy as np
from collections import deque, defaultdict
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import cv2
from scipy.spatial import distance
import random, copy, math
import numpy as np
from sys import path

path.append("./ultra")
from ultra.utils.common import (
    resize_im,
    get_closest_waypoint,
    compute_grad,
    get_path_to_goal,
    ego_social_safety,
    clip_angle_to_pi,
    rotate2d_vector,
)

import os

seed = 0
random.seed(seed)
num_lookahead = 100

# def action_adapter(agent.id):
#     agent.action_space_type
#     pass


class BaselineAdapter:
    def __init__(self):
        pass

    def observation_adapter(self, env_observation):
        ego_state = env_observation.ego_vehicle_state
        start = env_observation.ego_vehicle_state.mission.start
        goal = env_observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=env_observation.waypoint_paths, start=start
        )
        closest_wp, _ = get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=path,
            ego_position=ego_state.position,
            ego_heading=ego_state.heading,
        )
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        relative_goal_position = np.asarray(goal.position[0:2]) - np.asarray(
            ego_state.position[0:2]
        )
        relative_goal_position_rotated = rotate2d_vector(
            relative_goal_position, -ego_state.heading
        )

        state = dict(
            speed=ego_state.speed,
            relative_goal_position=relative_goal_position_rotated,
            distance_from_center=ego_dist_center,
            steering=ego_state.steering,
            angle_error=closest_wp.relative_heading(ego_state.heading),
            social_vehicles=env_observation.neighborhood_vehicle_states,
            road_speed=closest_wp.speed_limit,
            # ----------
            # dont normalize the following,
            start=start.position,
            goal=goal.position,
            heading=ego_state.heading,
            goal_path=path,
            ego_position=ego_state.position,
            waypoint_paths=env_observation.waypoint_paths,
        )
        ego = dict(
            position=ego_state.position,
            speed=ego_state.speed,
            steering=ego_state.steering,
            heading=ego_state.heading,
            dist_center=abs(ego_dist_center),
            start=start,
            goal=goal,
            path=path,
            closest_wp=closest_wp,
            events=env_observation.events,
        )
        return dict(state=state, ego=ego, env_observation=env_observation)

    def reward_adapter(self, observation, reward):
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
        ego_collision_reward = -10.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 100.0 if ego_events.reached_goal else 0.0
        ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        social_safety_reward = -0.02 if social_num_violations > 0 else 0
        ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
        ego_linear_jerk = -0.0001 * linear_jerk
        ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
        env_reward /= 10
        # DG: Different speed reward
        ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

        rewards = [
            # ego_goal_reward,
            # ego_collision_reward,
            # ego_off_road_reward,
            # ego_off_route_reward,
            # ego_wrong_way,
            ego_speed_reward,
            # ego_time_out,
            ego_dist_center_reward,
            ego_angle_error_reward,
            # ego_reached_goal,
            # ego_step_reward,
            env_reward,
            # ego_linear_jerk,
            # ego_angular_jerk,
            # ego_lat_speed,
            # ego_safety_reward,
            # social_safety_reward,
        ]
        return dict(
            reward=sum(rewards),
            log=dict(
                ego_social_safety_reward=ego_safety_reward + social_safety_reward,
                ego_num_violations=ego_num_violations,
                social_num_violations=social_num_violations,
                goal_dist=goal_dist,
                linear_jerk=0,
                angular_jerk=0,
            ),
        )


# TODO: move to policy
# ogm = self.get_ogm_from_obs(obs=env_observation)
# rgb = self.get_rgb_from_obs(obs=env_observation)
# if self.init:
#     self.ogm_stack = deque([ogm] * self.history, maxlen=self.history)
#     self.rgb_stack = deque([rgb] * self.history, maxlen=self.history)
# else:
#     self.ogm_stack.append(ogm)
#     self.rgb_stack.append(rgb)
# agent_ogm_state = np.concatenate(self.ogm_stack, axis=2).transpose([2, 0, 1])
# agent_rgb_state = np.concatenate(self.rgb_stack, axis=2).transpose([2, 0, 1])


# def get_ogm_from_obs(self, obs):
#     return resize_im(obs.occupancy_grid_map.data, self.ogm_size)

# def get_rgb_from_obs(self, obs, gray_scale=False):
#     if gray_scale:
#         gray_img = cv2.cvtColor(obs.top_down_rgb, cv2.COLOR_RGB2GRAY)
#         # plt.imsave('here.jpg', gray_img, cmap='gray')
#         gray_img = gray_img[:, :, np.newaxis]
#         return resize_im(gray_img, self.rgb_size)
#     return resize_im(obs.top_down_rgb, self.rgb_size)
# _INTERSECTION_OBSERVATION_SPACE = gym.spaces.Dict(
#     {
#         "ogm": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
#         "rgb": gym.spaces.Box(low=0.0, high=256.0, shape=(1,)),
#         "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
#         "angle_error": gym.spaces.Box(low=-180, high=180, shape=(1,)),
#         "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
#         "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
#         "relative_goal_position": gym.spaces.Box(
#             low=np.array([-1e10, -1e10]), high=np.array([1e10, 1e10]), dtype=np.float32
#         ),
#         "action": gym.spaces.Box(
#             low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
#         ),
#         "waypoints_lookahead": gym.spaces.Box(
#             low=np.array([-1e10]), high=np.array([1e10]), dtype=np.float32
#         ),
#         "social_vehicles": gym.spaces.Box(
#             low=np.array([-1e10, -1e10, -1e10, -1e10]),
#             high=np.array([1e10, 1e10, 1e10, 1e10]),
#             dtype=np.float32,
#         ),
#     }
# )

# goal_path = get_path_to_goal(goal=goal, paths=paths, start=start)
# goal_closest_wp, waypoints_lookahead = get_closest_waypoint(
#     env_observation=env_observation,
#     path=goal_path,
#     goal_pos=goal.position,
#     num_lookahead=self.num_lookahead,
# )
