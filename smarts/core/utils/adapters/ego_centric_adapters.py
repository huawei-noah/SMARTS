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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading
from smarts.core.observations import Observation, ViaPoint
from smarts.core.plan import PositionalGoal, Via
from smarts.core.utils.file import replace as _replace
from smarts.core.utils.math import (
    position_to_ego_frame,
    world_position_from_ego_frame,
    wrap_value,
)


def ego_centric_observation_adapter(obs: Observation, *args: Any, **kwargs: Any) -> Any:
    """An observation adapter that converts the observation to an ego-centric perspective."""

    position = obs.ego_vehicle_state.position
    heading = obs.ego_vehicle_state.heading

    def ego_frame_dynamics(v):
        return np.array([np.linalg.norm(v[:2]), 0, *v[2:]])  # point to X

    def transform(v):
        return position_to_ego_frame(v, position, heading)

    def adjust_heading(h):
        return wrap_value(h - heading, -math.pi, math.pi)

    nvs = obs.neighborhood_vehicle_states or []
    wpps = obs.waypoint_paths or []

    def _replace_via(via: Union[Via, ViaPoint]):
        return _replace(via, position=transform(via.position))

    vd = None
    if obs.via_data:
        rpvp = lambda vps: [_replace_via(vp) for vp in vps]
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
                heading=Heading(adjust_heading(wp.heading)),
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
            cam_obs.metadata, camera_position=(0, 0, 0), camera_heading_in_degrees=0
        ),
    )

    def _optional_replace_goal(goal):
        if isinstance(goal, PositionalGoal):
            return {"goal": _replace(goal, position=transform(tuple(goal.position)))}

        return {}

    def _replace_lidar(lidar):
        if len(lidar) == 0:
            return []
        return [
            [transform(hit_point) for hit_point in lidar[0]],
            lidar[1],
            [
                [transform(ray_start), transform(ray_end)]
                for ray_start, ray_end in lidar[2]
            ],
        ]

    return _replace(
        obs,
        ego_vehicle_state=_replace(
            obs.ego_vehicle_state,
            position=np.array([0, 0, 0]),
            heading=Heading(0),
            linear_velocity=ego_frame_dynamics(obs.ego_vehicle_state.linear_velocity),
            linear_acceleration=ego_frame_dynamics(
                obs.ego_vehicle_state.linear_acceleration
            ),
            linear_jerk=ego_frame_dynamics(obs.ego_vehicle_state.linear_jerk),
            mission=_replace(
                obs.ego_vehicle_state.mission,
                start=_replace(
                    obs.ego_vehicle_state.mission.start,
                    position=transform(
                        np.append(obs.ego_vehicle_state.mission.start.position, [0])
                    )[:2],
                    heading=adjust_heading(obs.ego_vehicle_state.mission.start.heading),
                ),
                via=tuple(
                    _replace_via(via) for via in obs.ego_vehicle_state.mission.via
                ),
                **_optional_replace_goal(obs.ego_vehicle_state.mission.goal),
                # TODO??: `entry_tactic.zone` zone.position?
            ),
        ),
        neighborhood_vehicle_states=[
            _replace(
                nv,
                position=transform(nv.position),
                heading=Heading(adjust_heading(nv.heading)),
            )
            for nv in nvs
        ],
        lidar_point_cloud=_replace_lidar(obs.lidar_point_cloud),
        waypoint_paths=replace_wps(wpps),
        drivable_area_grid_map=replace_metadata(obs.drivable_area_grid_map),
        occupancy_grid_map=replace_metadata(obs.occupancy_grid_map),
        top_down_rgb=replace_metadata(obs.top_down_rgb),
        road_waypoints=rwps,
        via_data=vd,
    )


def _egocentric_continuous_action_adapter(act: Tuple[float, float, float], _=None):
    return act


def _egocentric_actuator_dynamic_adapter(act: Tuple[float, float, float], _=None):
    return act


def _egocentric_lane_adapter(act: str, _=None):
    return act


def _egocentric_lane_with_continous_speed_adapter(act: Tuple[int, float], _=None):
    return act


def _trajectory_adaption(act, last_obs):
    new_pos = np.array(
        [
            world_position_from_ego_frame(
                [x, y, 0],
                last_obs.ego_vehicle_state.position,
                last_obs.ego_vehicle_state.heading,
            )[:2]
            for x, y in zip(*act[:2])
        ]
    ).T
    new_headings = np.array(
        [
            wrap_value(h + last_obs.ego_vehicle_state.heading, -math.pi, math.pi)
            for h in act[2]
        ]
    )
    return (*new_pos, new_headings, *act[3:])


def _egocentric_trajectory_adapter(
    act: Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
    last_obs: Optional[Observation] = None,
):
    if last_obs:
        return _trajectory_adaption(act, last_obs)
    return act


def _egocentric_trajectory_with_time_adapter(
    act: Tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ],
    last_obs: Optional[Observation] = None,
):
    if last_obs:
        return _trajectory_adaption(act, last_obs)
    return act


def _egocentric_mpc_adapter(
    act: Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
    last_obs: Optional[Observation] = None,
):
    if last_obs:
        return _trajectory_adaption(act, last_obs)
    return act


def _egocentric_target_pose_adapter(
    act: Tuple[float, float, float, float], last_obs: Optional[Observation] = None
):
    if last_obs:
        out_pos = world_position_from_ego_frame(
            np.append(act[:2], [0]),
            last_obs.ego_vehicle_state.position,
            last_obs.ego_vehicle_state.heading,
        )
        return np.array(
            [
                *out_pos[:2],
                wrap_value(
                    last_obs.ego_vehicle_state.heading + act[2], -math.pi, math.pi
                ),
                act[3],
            ]
        )
    return act


def _egocentric_multi_target_pose_adapter(
    act: Dict[str, Tuple[float, float, float, float]],
    last_obs: Optional[Observation] = None,
):
    assert ValueError(
        "Ego-centric assumes single vehicle and is ambiguous with multi-target-pose."
    )


def _egocentric_direct_adapter(
    act: Union[float, Tuple[float, float]], last_obs: Optional[Observation] = None
):
    return act


def _pair_adapters(
    ego_centric_observation_adapter: Callable[[Observation], Observation],
    ego_centric_action_adapter: Callable[[Any, Optional[Observation]], Any],
):
    """Wrapper that shares the state between both adapters."""
    last_obs = None

    def oa_wrapper(obs: Observation):
        nonlocal last_obs

        last_obs = obs  # Store the unmodified observation
        return ego_centric_observation_adapter(obs)

    def aa_wrapper(act: Any):
        nonlocal last_obs

        # Pass the last unmodified obs to the action for conversion purposes
        return ego_centric_action_adapter(act, last_obs)

    return oa_wrapper, aa_wrapper


def get_egocentric_adapters(action_space: ActionSpaceType):
    """Provides a set of adapters that share state information of the unmodified observation.
    This will allow the action adapter to automatically convert back to world space for SMARTS.
    Returns:
        (obs_adapter, action_adapter)
    """
    m = {
        ActionSpaceType.Continuous: _egocentric_continuous_action_adapter,
        ActionSpaceType.ActuatorDynamic: _egocentric_actuator_dynamic_adapter,
        ActionSpaceType.Lane: _egocentric_lane_adapter,
        ActionSpaceType.LaneWithContinuousSpeed: _egocentric_lane_with_continous_speed_adapter,
        ActionSpaceType.Trajectory: _egocentric_trajectory_adapter,
        ActionSpaceType.TrajectoryWithTime: _egocentric_trajectory_with_time_adapter,
        ActionSpaceType.MPC: _egocentric_mpc_adapter,
        ActionSpaceType.TargetPose: _egocentric_target_pose_adapter,
        ActionSpaceType.MultiTargetPose: _egocentric_multi_target_pose_adapter,
        ActionSpaceType.Direct: _egocentric_direct_adapter,
        ActionSpaceType.Empty: lambda _: None,
    }

    return _pair_adapters(ego_centric_observation_adapter, m.get(action_space))
