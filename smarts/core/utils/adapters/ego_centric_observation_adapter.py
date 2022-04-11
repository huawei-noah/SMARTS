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
from dataclasses import is_dataclass
from dataclasses import replace as dc_replace
from typing import Any

import numpy as np

from smarts.core.coordinates import Heading
from smarts.core.sensors import Observation
from smarts.core.utils.math import position_to_ego_frame


def _isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def _replace(obj: Any, **kwargs):
    if is_dataclass(obj):
        return dc_replace(obj, **kwargs)
    elif _isnamedtupleinstance(obj):
        return obj._replace(**kwargs)

    raise ValueError("Must be a namedtuple or dataclass.")


def ego_centric_observation_adapter(obs: Observation, *args: Any, **kwargs: Any) -> Any:
    """An observation adapter that converts the observation to an ego-centric perspective."""

    position = obs.ego_vehicle_state.position
    heading = obs.ego_vehicle_state.heading

    def ego_frame_dynamics(v):
        return np.array([np.linalg.norm(v[:2]), 0, *v[2:]])

    def transform(v):
        return position_to_ego_frame(v, position, heading)

    def adjust_heading(h):
        return h - heading

    nvs = obs.neighborhood_vehicle_states or []
    wpps = obs.waypoint_paths or []

    vd = None
    if obs.via_data:
        rpvp = lambda vps: [_replace(vp, position=transform(vp.position)) for vp in vps]
        vd = _replace(
            obs.via_data,
            near_via_points=rpvp(obs.via_data.near_via_points),
            hit_via_points=rpvp(obs.via_data.hit_via_points),
        )
    replace_wps = lambda lwps: [
        [
            _replace(
                wp,
                pos=transform(np.append(wp.pos, [0]))[:2],
                heading=adjust_heading(wp.heading),
            )
            for wp in wps
        ]
        for wps in lwps
    ]
    rwps = None
    if obs.road_waypoints:
        rwps = _replace(
            obs.road_waypoints,
            lanes={
                l_id: replace_wps(wps) for l_id, wps in obs.road_waypoints.lanes.items()
            },
        )

    replace_metadata = lambda cam_obs: _replace(
        cam_obs,
        metadata=_replace(
            cam_obs.metadata, camera_pos=(0, 0, 0), camera_heading_in_degrees=0
        ),
    )
    return _replace(
        obs,
        ego_vehicle_state=_replace(
            obs.ego_vehicle_state,
            position=np.array([0, 0, 0]),
            steering=Heading(0),
            linear_velocity=ego_frame_dynamics(obs.ego_vehicle_state.linear_velocity),
            linear_acceleration=ego_frame_dynamics(
                obs.ego_vehicle_state.linear_acceleration
            ),
            linear_jerk=ego_frame_dynamics(obs.ego_vehicle_state.linear_jerk),
        ),
        neighborhood_vehicle_states=[
            _replace(
                nv,
                position=transform(nv.position),
                heading=adjust_heading(nv.heading),
            )
            for nv in nvs
        ],
        waypoint_paths=replace_wps(wpps),
        drivable_area_grid_map=replace_metadata(obs.drivable_area_grid_map),
        occupancy_grid_map=replace_metadata(obs.occupancy_grid_map),
        top_down_rgb=replace_metadata(obs.top_down_rgb),
        road_waypoints=rwps,
        via_data=vd,
    )
