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
import torch
from ultra.utils.common import normalize_im, to_2d_action, to_3d_action
from collections.abc import Iterable
import numpy as np
from smarts.core.sensors import VehicleObservation
from ultra.baselines.common.social_vehicle_extraction import get_social_vehicles
from ultra.utils.common import (
    get_closest_waypoint,
    rotate2d_vector,
)
from ultra.scenarios.common.visualization import draw_intersection

identity_func = lambda x, *args, **kwargs: x


class StatePreprocessor:
    def __init__(self, preprocess_state_func, convert_action_func, state_description):
        self.preprocess_state_func = preprocess_state_func
        self.convert_action_func = convert_action_func
        self.state_description = state_description

    def __call__(
        self,
        state,
        social_capacity,
        observation_num_lookahead,
        social_vehicle_config,
        prev_action,
        draw=False,
        normalize=False,
        unsqueeze=False,
        device=None,
    ):
        return self.preprocess_state_func(
            state,
            self.state_description,
            observation_num_lookahead=observation_num_lookahead,
            social_capacity=social_capacity,
            normalize=normalize,
            unsqueeze=unsqueeze,
            device=device,
            convert_action_func=self.convert_action_func,
            social_vehicle_config=social_vehicle_config,
            prev_action=prev_action,
            draw=draw,
        )


def preprocess_state(
    state,
    state_description,
    convert_action_func,
    observation_num_lookahead,
    social_capacity,
    social_vehicle_config,
    prev_action,
    normalize=False,
    unsqueeze=False,
    device=None,
    draw=False,
):
    state = state.copy()
    images = {}
    for k in state_description["images"]:
        image = torch.from_numpy(state[k])
        image = image.unsqueeze(0) if unsqueeze else image
        image = image.to(device) if device else image
        image = normalize_im(image) if normalize else image
        images[k] = image

    if "action" in state:
        state["action"] = convert_action_func(state["action"])

    # -------------------------------------
    # filter lookaheads from goal_path
    _, lookahead_wps = get_closest_waypoint(
        num_lookahead=observation_num_lookahead,
        goal_path=state["goal_path"],
        ego_position=state["ego_position"],
        ego_heading=state["heading"],
    )
    state["waypoints_lookahead"] = np.hstack(lookahead_wps)

    # -------------------------------------
    # keep prev_action
    state["action"] = prev_action

    # -------------------------------------
    # normalize states and concat
    normalized = [
        _normalize(key, state[key]) for key in state_description["low_dim_states"]
    ]

    low_dim_states = [
        val if isinstance(val, Iterable) else np.asarray([val]).astype(np.float32)
        for val in normalized
    ]
    low_dim_states = torch.cat(
        [torch.from_numpy(e).float() for e in low_dim_states], dim=-1
    )
    low_dim_states = low_dim_states.unsqueeze(0) if unsqueeze else low_dim_states
    low_dim_states = low_dim_states.to(device) if device else low_dim_states

    # -------------------------------------
    # apply social vehicle encoder
    # only process if state is not encoded already
    state["social_vehicles"] = (
        get_social_vehicles(
            ego_vehicle_pos=state["ego_position"],
            ego_vehicle_heading=state["heading"],
            neighborhood_vehicles=state["social_vehicles"],
            social_vehicle_config=social_vehicle_config,
            waypoint_paths=state["waypoint_paths"],
        )
        if social_capacity > 0
        else []
    )
    # check if any social capacity is 0
    social_vehicle_dimension = state_description["social_vehicles"]
    social_vehicles = torch.empty(0, 0)

    if social_vehicle_dimension:
        social_vehicles = torch.from_numpy(np.asarray(state["social_vehicles"])).float()
        social_vehicles = social_vehicles.reshape((-1, social_vehicle_dimension))
    social_vehicles = social_vehicles.unsqueeze(0) if unsqueeze else social_vehicles
    social_vehicles = social_vehicles.to(device) if device else social_vehicles

    out = {
        "images": images,
        "low_dim_states": low_dim_states,
        "social_vehicles": social_vehicles,
    }
    return out


def get_state_description(
    social_vehicle_config, observation_waypoints_lookahead, action_size
):
    return {
        "images": {},
        "low_dim_states": {
            "speed": 1,
            "distance_from_center": 1,
            "steering": 1,
            "angle_error": 1,
            "relative_goal_position": 2,
            "action": int(action_size),  # 2
            "waypoints_lookahead": 2 * int(observation_waypoints_lookahead),
            "road_speed": 1,
        },
        "social_vehicles": int(social_vehicle_config["num_social_features"])
        if int(social_vehicle_config["social_capacity"]) > 0
        else 0,
    }


def _normalize(key, val):
    ref = {
        "speed": 30.0,
        "distance_from_center": 1.0,
        "steering": 3.14,  # radians
        "angle_error": 3.14,  # radians
        "relative_goal_position": 100.0,
        "action": 1.0,  # 2
        "waypoints_lookahead": 10.0,
        "road_speed": 30.0,
    }
    if key not in ref:
        return val
    return val / ref[key]


# all_waypoints = [
#     [linked_wp.pos[0], linked_wp.pos[1]] for linked_wp in state["goal_path"]
# ]
# if draw:
#     draw_intersection(
#         ego=state["ego_position"],
#         goal_path=state["goal_path"],
#         all_waypoints=all_waypoints,
#         step=step,
#         goal=state["goal"],
#         start=state["start"],
#         lookaheads=state["waypoints_lookahead"],
#         social_vehicle_states=state["social_vehicles"],
#         finished_vehicles=[],
#     )
