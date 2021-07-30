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
# The author of this file is: https://github.com/mg2015started

import heapq

import numpy as np

from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType


def observation_adapter(env_obs):
    ego_feature_num = 4
    npc_feature_num = 5
    near_npc_number = 5
    mask_size = near_npc_number + 1
    env_state_size = ego_feature_num + near_npc_number * npc_feature_num
    # get ego state
    ego_states = env_obs.ego_vehicle_state
    ego_x = ego_states.position[0]
    ego_y = ego_states.position[1]
    ego_loc = ego_states.position[0:2]
    ego_mission = ego_states.mission
    ego_yaw = ego_states.heading
    ego_speed = ego_states.speed
    # update neighbor vehicle list
    detect_range = 37.5
    veh_within_detect_range_list = []
    for index, vehicle_state in enumerate(env_obs.neighborhood_vehicle_states):
        npc_loc = vehicle_state.position[0:2]
        distance = np.linalg.norm(npc_loc - ego_loc)
        if distance < detect_range:
            add_dict = {"vehicle_state": vehicle_state, "distance": distance}
            veh_within_detect_range_list.append(add_dict)

    r_veh_list = []
    ir_veh_list = []
    # Get relavent npc vehicle
    for veh_dic in veh_within_detect_range_list:
        npc_x = veh_dic["vehicle_state"].position[0]
        npc_y = veh_dic["vehicle_state"].position[1]
        npc_yaw = veh_dic["vehicle_state"].heading

        distance = veh_dic["distance"]
        y_relative = (npc_y - ego_y) * np.cos(ego_yaw) - (npc_x - ego_x) * np.sin(
            ego_yaw
        )

        yaw_relative = npc_yaw - ego_yaw

        if y_relative < -5 or (yaw_relative < 0.1 and distance > 10):
            ir_veh_list.append(veh_dic)
        else:
            r_veh_list.append(veh_dic)

    # sort the vehicles according to their distance
    _near_npc = heapq.nsmallest(
        near_npc_number, r_veh_list, key=lambda s: s["distance"]
    )
    distance_list = []
    for i in range(len(_near_npc)):
        distance_list.append(_near_npc[i]["distance"])
    # print('nearest veh:', distance_list)
    r_npc_list = [x["vehicle_state"] for x in _near_npc]
    ir_npc_list = [x["vehicle_state"] for x in ir_veh_list]

    # get environment state
    env_state = []
    if ego_states.edge_id == "edge-south-SN":  # start lane
        ego_pos_flag = [1, 0, 0]
    elif "junction" in ego_states.edge_id:  # junction
        ego_pos_flag = [0, 1, 0]
    else:  # goal lane
        ego_pos_flag = [0, 0, 1]

    ego_state = ego_pos_flag + [ego_speed]
    # print(ego_states.speed)
    env_state += ego_state
    # print('step')
    for veh_state in r_npc_list:
        # coordinates relative to ego
        npc_x = veh_state.position[0]
        npc_y = veh_state.position[1]
        npc_yaw = veh_state.heading
        x_relative = (npc_y - ego_y) * np.sin(ego_yaw) + (npc_x - ego_x) * np.cos(
            ego_yaw
        )
        y_relative = (npc_y - ego_y) * np.cos(ego_yaw) - (npc_x - ego_x) * np.sin(
            ego_yaw
        )
        # yaw relative to ego
        delta_yaw = npc_yaw - ego_yaw
        # speed
        npc_speed = veh_state.speed
        # state representation for RL
        # print(np.linalg.norm(np.array([x_relative, y_relative])))
        npc_state = [
            x_relative,
            y_relative,
            npc_speed,
            np.cos(delta_yaw),
            np.sin(delta_yaw),
        ]
        # print(ego_x, npc_x, x_relative, ego_y, npc_y, y_relative)

        # intergrate states
        env_state += npc_state

    # get aux state, whichs include task vector & vehicle mask
    mask = list(np.ones(mask_size))
    if len(env_state) < env_state_size:
        zero_padding_num = int((env_state_size - len(env_state)) / npc_feature_num)
        for _ in range(zero_padding_num):
            mask.pop()
        for _ in range(zero_padding_num):
            mask.append(0)
        while len(env_state) < env_state_size:
            env_state.append(0)

    goal_x = ego_mission.goal.position[0]
    if goal_x == 127.6:
        task = [1, 0, 0, 1]
    elif goal_x == 151.6:
        task = [0, 1, 0, 1]
    elif goal_x == 172.4:
        task = [0, 0, 1, 1]
    aux_state = mask + task

    # final state_mask
    total_state = np.array(env_state + aux_state, dtype=np.float32)
    # print(state_mask, state_mask.shape)
    return total_state


def action_adapter(action):
    target_speed = np.clip(action[0] - action[1] / 4, 0, 1)
    target_speed = target_speed * 12

    agent_action = [target_speed, int(0)]
    return agent_action


def reward_adapter(env_obs, reward):
    # set task vector
    goal_x = env_obs.ego_vehicle_state.mission.goal.position[0]
    if goal_x == 127.6:
        task = [1, 0, 0, 1]
    elif goal_x == 151.6:
        task = [0, 1, 0, 1]
    elif goal_x == 172.4:
        task = [0, 0, 1, 1]

    # compute reward vector
    reward_c = 0.0
    reward_s = 0.0
    ego_events = env_obs.events

    # checking
    collision = len(ego_events.collisions) > 0  # check collision
    time_exceed = ego_events.reached_max_episode_steps  # check time exceeds
    reach_goal = ego_events.reached_goal
    # penalty
    reward_c += -0.3  # step cost
    if collision:
        print("collision:", ego_events.collisions)
        print("nearest veh:", observation_adapter(env_obs)[4:9])
        print("Failure. Ego vehicle collides with npc vehicle.")
        reward_s += -650
    elif time_exceed:
        print("nearest veh:", observation_adapter(env_obs)[4:9])
        print("Failure. Time exceed.")
        reward_c += -50
    # reward
    else:
        if reach_goal:
            print("nearest veh:", observation_adapter(env_obs)[4:9])
            print("Success. Ego vehicle reached goal.")
            reward_s += 30

    # reward vector for multi task
    reward = [i * reward_s for i in task[0:3]] + [reward_c]

    return reward


def get_aux_info(env_obs):
    ego_events = env_obs.events
    collision = len(ego_events.collisions) > 0  # check collision
    time_exceed = ego_events.reached_max_episode_steps  # check time exceeds
    reach_goal = ego_events.reached_goal
    if collision:
        aux_info = "collision"
    elif time_exceed:
        aux_info = "time_exceed"
    elif reach_goal:
        aux_info = "success"
    else:
        aux_info = "running"
    return aux_info


cross_interface = AgentInterface(
    max_episode_steps=500,
    neighborhood_vehicles=True,
    waypoints=True,
    action=ActionSpaceType.LaneWithContinuousSpeed,
)
