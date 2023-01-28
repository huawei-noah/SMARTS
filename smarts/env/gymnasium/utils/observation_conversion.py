# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import warnings
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from cached_property import cached_property

from smarts.core.agent_interface import AgentInterface
from smarts.core.events import Events
from smarts.core.observations import Observation, SignalObservation, VehicleObservation
from smarts.core.plan import Mission
from smarts.core.road_map import Waypoint

_LIDAR_SHP = 300
_NEIGHBOR_SHP = 10
_WAYPOINT_SHP = (4, 20)
_SIGNALS_SHP = (3,)
_POSITION_SHP = (3,)
_WAYPOINT_NAME_LIMIT = 50
_TEXT_PAD_CHAR = " "
_WAYPOINT_CHAR_SET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=+.,;\"' "
)

_VEC3_SIGNED_FLOAT32_SPACE = gym.spaces.Box(
    low=-1e10, high=1e10, shape=(3,), dtype=np.float32
)
_VEC3_UNSIGNED_FLOAT32_SPACE = gym.spaces.Box(
    low=0, high=1e10, shape=(3,), dtype=np.float32
)
_VEC3_SIGNED_FLOAT64_SPACE = gym.spaces.Box(
    low=-1e10, high=1e10, shape=(3,), dtype=np.float64
)
_SIGNED_FLOAT32_SPACE = gym.spaces.Box(low=-1e10, high=1e10, shape=(), dtype=np.float32)
_UNSIGNED_FLOAT32_SPACE = gym.spaces.Box(low=0, high=1e10, shape=(), dtype=np.float32)
_SIGNED_RADIANS_FLOAT32_SPACE = gym.spaces.Box(
    low=-math.pi, high=math.pi, shape=(), dtype=np.float32
)
_UNSIGNED_RADIANS_FLOAT32_SPACE = gym.spaces.Box(
    low=0, high=2 * math.pi, shape=(), dtype=np.float32
)
_UNSIGNED_INT8_SPACE = gym.spaces.Box(low=0, high=127, shape=(), dtype=np.int8)
_DISCRETE2_SPACE = gym.spaces.Discrete(n=2)
_LANE_ID_SPACE = gym.spaces.Text(_WAYPOINT_NAME_LIMIT, charset=_WAYPOINT_CHAR_SET)


def _format_lane_id(lane_id: str):
    lane_name = lane_id.ljust(_WAYPOINT_NAME_LIMIT, _TEXT_PAD_CHAR)
    if len(lane_name) > _WAYPOINT_NAME_LIMIT:
        warnings.warn(
            f"Lane named `{lane_name}` is more than "
            f"`{_WAYPOINT_NAME_LIMIT}` characters long. It will be truncated "
            "and may cause unintended issues with navigation and lane identification."
        )
    return lane_name[:_WAYPOINT_NAME_LIMIT]


def _format_mission(mission: Mission):
    if hasattr(mission.goal, "position"):
        goal_pos = np.array(mission.goal.position, dtype=np.float64)
    else:
        goal_pos = np.zeros((3,), dtype=np.float64)

    return {"goal_position": goal_pos}


def _format_waypoint_paths(waypoint_paths: List[List[Waypoint]]):
    # Truncate all paths to be of the same length
    min_len = min(map(len, waypoint_paths))
    trunc_paths = list(map(lambda x: x[:min_len], waypoint_paths))

    des_shp = _WAYPOINT_SHP
    rcv_shp = (len(trunc_paths), len(trunc_paths[0]))
    pad_shp = [0 if des - rcv < 0 else des - rcv for des, rcv in zip(des_shp, rcv_shp)]

    def extract_elem(waypoint: Waypoint):
        return (
            waypoint.lane_id,
            waypoint.heading,
            waypoint.lane_index,
            waypoint.lane_offset,
            waypoint.lane_width,
            waypoint.pos,
            waypoint.speed_limit,
        )

    paths = [
        map(extract_elem, path[: des_shp[1]]) for path in trunc_paths[: des_shp[0]]
    ]
    lane_id, heading, lane_index, lane_offset, lane_width, pos, speed_limit = zip(
        *[zip(*path) for path in paths]
    )

    # # TODO MTA: Add padded lane id
    # lane_id = ((l_id.ljust(_WAYPOINT_NAME_LIMIT, _TEXT_PAD_CHAR)[:_WAYPOINT_NAME_LIMIT] for l_id in s_lane_id) for s_lane_id in lane_id)
    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    lane_offset = np.array(lane_offset, dtype=np.float32)
    lane_width = np.array(lane_width, dtype=np.float32)
    pos = np.array(pos, dtype=np.float64)
    speed_limit = np.array(speed_limit, dtype=np.float32)

    # fmt: off
    # # TODO MTA: Add padded lane id
    heading = np.pad(heading, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_offset = np.pad(lane_offset, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_width = np.pad(lane_width, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp[0]),(0,pad_shp[1]),(0,1)), mode='constant', constant_values=0)
    speed_limit = np.pad(speed_limit, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    # fmt: on

    return {
        # # TODO MTA: Add padded lane id
        # "lane_id": lane_id,
        "heading": heading,
        "lane_index": lane_index,
        "lane_offset": lane_offset,
        "lane_width": lane_width,
        "position": pos,
        "speed_limit": speed_limit,
    }


def _format_signals(signals: List[SignalObservation]):
    des_shp = _SIGNALS_SHP
    rcv_shp = len(signals)
    pad_shp = max(0, des_shp[0] - rcv_shp)

    if rcv_shp == 0:
        return {
            "state": np.zeros(des_shp, dtype=np.int8),
            "stop_point": np.zeros(des_shp + (2,), dtype=np.float64),
            "last_changed": np.zeros(des_shp, dtype=np.float32),
        }

    signals = [
        (signal.state, signal.stop_point, signal.last_changed)
        for signal in signals[: des_shp[0]]
    ]
    state, stop_point, last_changed = zip(*signals)

    # fmt: off
    state = np.pad(state, ((0, pad_shp)), mode='constant', constant_values=0)
    stop_point = np.pad(state, ((0, pad_shp), (0, 0)), mode='constant', constant_values=0)
    last_changed = np.pad(last_changed, ((0, pad_shp)), mode='constant', constant_values=0)
    # fmt: on

    return {
        "state": state,
        "stop_point": stop_point,
        "last_changed": last_changed,
    }


def _format_neighborhood_vehicle_states(
    neighborhood_vehicle_states: List[VehicleObservation],
):
    ## TODO MTA: Add in the vehicle ids
    des_shp = _NEIGHBOR_SHP
    rcv_shp = len(neighborhood_vehicle_states)
    pad_shp = 0 if des_shp - rcv_shp < 0 else des_shp - rcv_shp

    if rcv_shp == 0:
        return {
            "box": np.zeros((des_shp, 3), dtype=np.float32),
            "heading": np.zeros((des_shp,), dtype=np.float32),
            "lane_index": np.zeros((des_shp,), dtype=np.int8),
            "position": np.zeros((des_shp, 3), dtype=np.float64),
            "speed": np.zeros((des_shp,), dtype=np.float32),
        }

    neighborhood_vehicle_states = [
        (
            nghb.bounding_box.as_lwh,
            nghb.heading,
            nghb.lane_index,
            nghb.position,
            nghb.speed,
        )
        for nghb in neighborhood_vehicle_states[:des_shp]
    ]
    box, heading, lane_index, pos, speed = zip(*neighborhood_vehicle_states)

    box = np.array(box, dtype=np.float32)
    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    pos = np.array(pos, dtype=np.float64)
    speed = np.array(speed, dtype=np.float32)

    # fmt: off
    box = np.pad(box, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    heading = np.pad(heading, ((0,pad_shp)), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp)), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    speed = np.pad(speed, ((0,pad_shp)), mode='constant', constant_values=0)
    # fmt: on

    return {
        ## TODO MTA: Add in the vehicle ids
        # "vehicle_id": vehicle_id,
        "box": box,
        "heading": heading,
        "lane_index": lane_index,
        "position": pos,
        "speed": speed,
    }


def _format_lidar(
    lidar_point_cloud: Optional[
        Tuple[List[np.ndarray], List[bool], List[Tuple[np.ndarray, np.ndarray]]]
    ]
):
    # # MTA TODO: add lidar configuration like following:
    # sensor_params = self._agent_interface.lidar_point_cloud.sensor_params
    # n_rays = int(
    #     (sensor_params.end_angle - sensor_params.start_angle)
    #     / sensor_params.angle_resolution
    # )
    des_shp = _LIDAR_SHP
    hit = np.array(lidar_point_cloud[1], dtype=np.int8)
    point_cloud = np.array(lidar_point_cloud[0], dtype=np.float64)
    point_cloud = np.nan_to_num(
        point_cloud,
        copy=False,
        nan=np.float64(0),
        posinf=np.float64(0),
        neginf=np.float64(0),
    )
    ray_origin, ray_vector = zip(*(lidar_point_cloud[2]))
    ray_origin = np.array(ray_origin, np.float64)
    ray_vector = np.array(ray_vector, np.float64)

    try:
        assert hit.shape == (des_shp,)
        assert point_cloud.shape == (des_shp, 3)
        assert ray_origin.shape == (des_shp, 3)
        assert ray_vector.shape == (des_shp, 3)
    except Exception as exc:
        raise Exception("Internal Error: Mismatched lidar point cloud shape.") from exc

    return {
        "hit": hit,
        "point_cloud": point_cloud,
        "ray_origin": ray_origin,
        "ray_vector": ray_vector,
    }


class BaseSpaceFormat:
    """Defines the base interface for an observation formatter."""

    def format(self, obs: Observation):
        """Selects and formats the given observation to get a value that matches the :attr:`space`."""
        raise NotImplementedError()

    def active(self, agent_interface: AgentInterface) -> bool:
        """If this formatting is active and should be included in the output."""
        raise NotImplementedError()

    @property
    def name(self):
        """The name that should represent this observation space in heirachy."""
        raise NotImplementedError()

    @property
    def space(self):
        """The observation space this should format the smarts observation to match."""
        raise NotImplementedError()

    def __call__(self, agent_interface: AgentInterface) -> "BaseSpaceFormat":
        """The observation space this should format the smarts observation to match."""
        raise NotImplementedError()


class StandardSpaceFormat(BaseSpaceFormat):
    """A formatter that is generated by configuration. This is immutable."""

    def __init__(
        self,
        formatting_func: Callable[[Observation], Dict[str, Any]],
        active_func: Callable[[AgentInterface], bool],
        name: str,
        space: gym.Space,
    ) -> None:
        self._formatting_func = formatting_func
        self._active_func = active_func
        self._name = name
        self._space = space

    def format(self, obs: Observation):
        """Selects and formats the given observation to get a value that matches the :attr:`space`."""
        return self._formatting_func(obs)

    def active(self, agent_interface: AgentInterface) -> bool:
        """If this formatting is active and should be included in the output."""
        return self._active_func(agent_interface)

    @property
    def name(self):
        """The name that should represent this observation space in heirachy."""
        return self._name

    @property
    def space(self):
        """The observation space this should format the smarts observation to match."""
        return self._space

    def __call__(self, agent_interface: AgentInterface) -> BaseSpaceFormat:
        return self


class StandardConfigurableSpaceFormat(BaseSpaceFormat):
    """A formatter that defers agent interface configuration."""

    def __init__(
        self,
        formatting_func: Callable[[Observation], Dict[str, Any]],
        active_func: Callable[[AgentInterface], bool],
        name: str,
        space_func: Callable[[AgentInterface], gym.Space],
        *,
        _agent_interface: Optional[AgentInterface] = None,
    ) -> None:
        self._formatting_func = formatting_func
        self._active_func = active_func
        self._name = name
        self._space_func = space_func
        self._agent_interface = _agent_interface

    def format(self, obs: Observation):
        """Selects and formats the given observation to get a value that matches the :attr:`space`."""
        return self._formatting_func(obs)

    def active(self, agent_interface: AgentInterface) -> bool:
        """If this formatting is active and should be included in the output."""
        return self._active_func(agent_interface)

    @property
    def name(self):
        """The name that should represent this observation space in heirachy."""
        return self._name

    @property
    def space(self):
        """The observation space this should format the smarts observation to match."""
        assert (
            self._agent_interface is not None
        ), "Agent interface must be applied to call this method."
        return self._space_func(self._agent_interface)

    def __call__(self, agent_interface: AgentInterface) -> BaseSpaceFormat:
        return type(self)(
            self._formatting_func,
            self._active_func,
            self._name,
            self._space_func,
            _agent_interface=agent_interface,
        )


class StandardCompoundSpaceFormat(BaseSpaceFormat):
    """A compound formatter that defers agent interface configuration."""

    def __init__(
        self,
        space_generators: List[Callable[[AgentInterface], BaseSpaceFormat]],
        active_func: Callable[[AgentInterface], bool],
        name: str,
        *,
        _spaces: Optional[List[BaseSpaceFormat]] = None,
    ) -> None:
        self._space_generators = space_generators
        self._spaces = _spaces or []
        self._active_func = active_func
        self._name = name

    def format(self, obs: Observation):
        return {s.name: s.format(obs) for s in self._spaces}

    def active(self, agent_interface: AgentInterface) -> bool:
        """If this formatting is active and should be included in the output."""
        return self._active_func(agent_interface)

    @property
    def name(self):
        """The name that should represent this observation space in heirachy."""
        return self._name

    @property
    def space(self):
        return gym.spaces.Dict({s.name: s.space for s in self._spaces})

    def __call__(self, agent_interface: AgentInterface) -> BaseSpaceFormat:
        _spaces = [space(agent_interface) for space in self._space_generators]
        return type(self)(
            [],
            self._active_func,
            self._name,
            _spaces=[space for space in _spaces if space.active(agent_interface)],
        )


class Image8BSpaceFormat(BaseSpaceFormat):
    """Defines a observation formatter which is an 8-bit image."""

    def __init__(self, dimensions, layers: int) -> None:
        self._dimensions = dimensions
        self._colors = layers

    @property
    def space(self):
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._dimensions.height, self._dimensions.width, self._colors),
            dtype=np.uint8,
        )

    def __call__(self, agent_interface: AgentInterface) -> BaseSpaceFormat:
        return self


ego_box_space_format = StandardSpaceFormat(
    lambda obs: np.array(obs.ego_vehicle_state.bounding_box.as_lwh).astype(np.float32),
    lambda _: True,
    "box",
    _VEC3_UNSIGNED_FLOAT32_SPACE,
)

ego_heading_space_format = StandardSpaceFormat(
    lambda obs: np.float32(obs.ego_vehicle_state.heading),
    lambda _: True,
    "heading",
    _SIGNED_RADIANS_FLOAT32_SPACE,
)

ego_lane_index_space_format = StandardSpaceFormat(
    lambda obs: np.int8(obs.ego_vehicle_state.lane_index),
    lambda _: True,
    "lane_index",
    _UNSIGNED_INT8_SPACE,
)

ego_linear_velocity_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.linear_velocity.astype(np.float32),
    lambda _: True,
    "linear_velocity",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

ego_angular_velocity_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.angular_velocity.astype(np.float32),
    lambda _: True,
    "angular_velocity",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

ego_position_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.position.astype(np.float64),
    lambda _: True,
    "position",
    _VEC3_SIGNED_FLOAT64_SPACE,
)

ego_speed_space_format = StandardSpaceFormat(
    lambda obs: np.float32(obs.ego_vehicle_state.speed),
    lambda _: True,
    "speed",
    _UNSIGNED_FLOAT32_SPACE,
)


ego_lane_id_space_format = StandardSpaceFormat(
    lambda obs: _format_lane_id(obs.ego_vehicle_state.lane_id),
    lambda _: True,
    "lane_id",
    _LANE_ID_SPACE,
)


ego_steering_space_format = StandardSpaceFormat(
    lambda obs: np.float32(obs.ego_vehicle_state.steering),
    lambda _: True,
    "steering",
    _SIGNED_RADIANS_FLOAT32_SPACE,
)

ego_yaw_rate_space_format = StandardSpaceFormat(
    lambda obs: np.float32(obs.ego_vehicle_state.yaw_rate),
    lambda _: True,
    "yaw_rate",
    _UNSIGNED_RADIANS_FLOAT32_SPACE,
)

ego_angular_acceleration_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.angular_acceleration.astype(np.float32),
    lambda agent_interface: bool(agent_interface.accelerometer),
    "angular_acceleration",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

ego_angular_jerk_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.angular_jerk.astype(np.float32),
    lambda agent_interface: bool(agent_interface.accelerometer),
    "angular_jerk",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

ego_linear_acceleration_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.linear_acceleration.astype(np.float32),
    lambda agent_interface: bool(agent_interface.accelerometer),
    "linear_acceleration",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

ego_linear_jerk_space_format = StandardSpaceFormat(
    lambda obs: obs.ego_vehicle_state.linear_jerk.astype(np.float32),
    lambda agent_interface: bool(agent_interface.accelerometer),
    "linear_jerk",
    _VEC3_SIGNED_FLOAT32_SPACE,
)

mission_space_format = StandardSpaceFormat(
    lambda obs: _format_mission(obs.ego_vehicle_state.mission),
    lambda _: True,
    "mission",
    gym.spaces.Dict(
        {
            "goal_position": gym.spaces.Box(
                low=-1e10, high=1e10, shape=_POSITION_SHP, dtype=np.float64
            )
        }
    ),
)

distance_travelled_space_format = StandardSpaceFormat(
    lambda obs: np.float32(obs.distance_travelled),
    lambda _: True,
    "distance_travelled",
    _SIGNED_FLOAT32_SPACE,
)


events_agents_alive_done_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.agents_alive_done),
    lambda _: True,
    "agents_alive_done",
    _DISCRETE2_SPACE,
)

events_collisions_space_format = StandardSpaceFormat(
    lambda obs: np.int64(len(obs.events.collisions) > 0),
    lambda _: True,
    "collisions",
    _DISCRETE2_SPACE,
)

events_not_moving_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.not_moving),
    lambda _: True,
    "not_moving",
    _DISCRETE2_SPACE,
)

events_off_road_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.off_road),
    lambda _: True,
    "off_road",
    _DISCRETE2_SPACE,
)

events_off_route_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.off_route),
    lambda _: True,
    "off_route",
    _DISCRETE2_SPACE,
)

events_on_shoulder_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.on_shoulder),
    lambda _: True,
    "on_shoulder",
    _DISCRETE2_SPACE,
)

events_reached_goal_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.reached_goal),
    lambda _: True,
    "reached_goal",
    _DISCRETE2_SPACE,
)

events_reached_max_episode_steps_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.reached_max_episode_steps),
    lambda _: True,
    "reached_max_episode_steps",
    _DISCRETE2_SPACE,
)

events_wrong_way_space_format = StandardSpaceFormat(
    lambda obs: np.int64(obs.events.wrong_way),
    lambda _: True,
    "wrong_way",
    _DISCRETE2_SPACE,
)


class DrivableAreaGridMapSpaceFormat(Image8BSpaceFormat):
    """Formats for `obs.drivable_area_grid_map`."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        super().__init__(dimensions=agent_interface.drivable_area_grid_map, layers=1)

    def format(self, obs: Observation):
        return obs.drivable_area_grid_map.data.astype(np.uint8)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.drivable_area_grid_map)

    @property
    def name(self):
        return "drivable_area_grid_map"


lidar_point_cloud_space_format = StandardConfigurableSpaceFormat(
    lambda obs: _format_lidar(obs.lidar_point_cloud),
    lambda agent_interface: bool(agent_interface.lidar_point_cloud),
    "lidar_point_cloud",
    # MTA TODO: add lidar configuration
    lambda _: gym.spaces.Dict(
        {
            "hit": gym.spaces.MultiBinary(_LIDAR_SHP),
            "point_cloud": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_LIDAR_SHP, 3), dtype=np.float64
            ),
            "ray_origin": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_LIDAR_SHP, 3), dtype=np.float64
            ),
            "ray_vector": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_LIDAR_SHP, 3), dtype=np.float64
            ),
        }
    ),
)


neighborhood_vehicle_states_space_format = StandardSpaceFormat(
    lambda obs: _format_neighborhood_vehicle_states(obs.neighborhood_vehicle_states),
    lambda agent_interface: bool(agent_interface.neighborhood_vehicle_states),
    "neighborhood_vehicle_states",
    # MTA TODO: add lidar configuration
    gym.spaces.Dict(
        {
            "box": gym.spaces.Box(
                low=0, high=1e10, shape=(_NEIGHBOR_SHP, 3), dtype=np.float32
            ),
            "heading": gym.spaces.Box(
                low=-math.pi, high=math.pi, shape=(_NEIGHBOR_SHP,), dtype=np.float32
            ),
            "lane_index": gym.spaces.Box(
                low=0, high=127, shape=(_NEIGHBOR_SHP,), dtype=np.int8
            ),
            "position": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_NEIGHBOR_SHP, 3), dtype=np.float64
            ),
            "speed": gym.spaces.Box(
                low=0, high=1e10, shape=(_NEIGHBOR_SHP,), dtype=np.float32
            ),
        }
    ),
)


class OccupancyGridMapSpaceFormat(Image8BSpaceFormat):
    """Formats for `obs.occupancy_grid_map`."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        super().__init__(dimensions=agent_interface.occupancy_grid_map, layers=1)

    def format(self, obs: Observation):
        return obs.occupancy_grid_map.data.astype(np.uint8)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.occupancy_grid_map)

    @property
    def name(self):
        return "occupancy_grid_map"


class TopDownRGBSpaceFormat(Image8BSpaceFormat):
    """Formats for `obs.top_down_rgb`."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        super().__init__(dimensions=agent_interface.top_down_rgb, layers=3)

    def format(self, obs: Observation):
        return obs.top_down_rgb.data.astype(np.uint8)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.top_down_rgb)

    @property
    def name(self):
        return "top_down_rgb"


waypoint_paths_space_format = StandardSpaceFormat(
    lambda obs: _format_waypoint_paths(obs.waypoint_paths),
    lambda agent_interface: bool(agent_interface.waypoint_paths),
    "waypoint_paths",
    gym.spaces.Dict(
        {
            # # TODO MTA: Add padded lane id
            # "lane_id": gym.spaces.Tuple(
            #     (
            #         gym.spaces.Tuple(
            #             gym.spaces.Text(
            #                 _WAYPOINT_NAME_LIMIT,
            #                 charset=frozenset(
            #                     "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=+"
            #                 ),
            #             ) for _ in range(_WAYPOINT_SHP[1])
            #         )
            #         for _ in range(_WAYPOINT_SHP[0])
            #     )
            # ),
            "heading": gym.spaces.Box(
                low=-math.pi, high=math.pi, shape=_WAYPOINT_SHP, dtype=np.float32
            ),
            "lane_index": gym.spaces.Box(
                low=0, high=127, shape=_WAYPOINT_SHP, dtype=np.int8
            ),
            "lane_offset": gym.spaces.Box(
                low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32
            ),
            "lane_width": gym.spaces.Box(
                low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32
            ),
            "position": gym.spaces.Box(
                low=-1e10, high=1e10, shape=_WAYPOINT_SHP + (3,), dtype=np.float64
            ),
            "speed_limit": gym.spaces.Box(
                low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32
            ),
        }
    ),
)


signals_space_format = StandardSpaceFormat(
    lambda obs: _format_signals(obs.signals),
    lambda agent_interface: bool(agent_interface.signals),
    "signals",
    gym.spaces.Dict(
        {
            "state": gym.spaces.Box(low=0, high=32, shape=_SIGNALS_SHP, dtype=np.int8),
            "stop_point": gym.spaces.Box(
                low=-1e10, high=1e10, shape=_SIGNALS_SHP + (2,), dtype=np.float64
            ),
            "last_changed": gym.spaces.Box(
                low=0, high=1e10, shape=_SIGNALS_SHP, dtype=np.float32
            ),
        }
    ),
)


enabled_space_format = StandardSpaceFormat(
    lambda _: np.int64(True),
    lambda _: True,
    "active",
    _DISCRETE2_SPACE,
)

ego_vehicle_state_space_format = StandardCompoundSpaceFormat(
    space_generators=[
        # required
        ego_angular_velocity_space_format,
        ego_box_space_format,
        ego_heading_space_format,
        ego_lane_id_space_format,
        ego_lane_index_space_format,
        ego_linear_velocity_space_format,
        ego_position_space_format,
        ego_speed_space_format,
        ego_steering_space_format,
        ego_yaw_rate_space_format,
        mission_space_format,
        # optional
        ego_angular_acceleration_space_format,
        ego_angular_jerk_space_format,
        ego_linear_acceleration_space_format,
        ego_linear_jerk_space_format,
    ],
    active_func=lambda _: True,
    name="ego_vehicle_state",
)


events_space_format = StandardCompoundSpaceFormat(
    space_generators=[
        events_agents_alive_done_space_format,
        events_collisions_space_format,
        events_not_moving_space_format,
        events_off_road_space_format,
        events_off_route_space_format,
        events_on_shoulder_space_format,
        events_reached_goal_space_format,
        events_reached_max_episode_steps_space_format,
        events_wrong_way_space_format,
    ],
    active_func=lambda _: True,
    name="events",
)


observation_space_format = StandardCompoundSpaceFormat(
    space_generators=[
        enabled_space_format,
        distance_travelled_space_format,
        ego_vehicle_state_space_format,
        events_space_format,
        DrivableAreaGridMapSpaceFormat,
        lidar_point_cloud_space_format,
        neighborhood_vehicle_states_space_format,
        OccupancyGridMapSpaceFormat,
        TopDownRGBSpaceFormat,
        waypoint_paths_space_format,
        signals_space_format,
    ],
    active_func=lambda _: True,
    name="observation",
)


class ObservationOptions(IntEnum):
    """Defines the options for how the formatting matches the observation space."""

    multi_agent = 0
    """Observation partially matches observation space. Only active agents are included."""
    full = 1
    """Observation fully matches observation space. Inactive and active agents are included."""
    unformatted = 2
    """Observation is the original unformatted observation. The observation will not match the
    observation space."""
    default = 0
    """Defaults to :attr:`multi_agent`."""


class ObservationSpacesFormatter:
    """Formats a smarts observation to fixed sized object.

    Observations in numpy array format, suitable for vectorized processing.

    For each agent id::

        obs = dict({

            If the agent is active.
            "active": 1 if agent is active in smarts, else 0

            Total distance travelled in meters.
            "distance_travelled": np.float32

            Ego vehicle state, with the following attributes.
            "ego_vehicle_state": dict({
                "angular_acceleration":
                    Angular acceleration vector. Requires `accelerometer` attribute
                    enabled in AgentInterface, else absent. shape=(3,). dtype=np.float32.
                "angular_jerk":
                    Angular jerk vector. Requires `accelerometer` attribute enabled in
                    AgentInterface, else absent. shape=(3,). dtype=np.float32.
                "angular_velocity":
                    Angular velocity vector. shape=(3,). dtype=np.float32).
                "box":
                    Length, width, and height of the vehicle bounding box. shape=(3,).
                    dtype=np.float32.
                "heading":
                    Vehicle heading in radians [-pi, pi]. dtype=np.float32.
                "lane_id":
                    The ID of the lane that the vehicle is on.
                "lane_index":
                    Vehicle's lane number. Rightmost lane has index 0 and increases
                    towards left. dtype=np.int8.
                "linear_acceleration":
                    Vehicle acceleration in x, y, and z axes. Requires `accelerometer`
                    attribute enabled in AgentInterface, else absent. shape=(3,).
                    dtype=np.float32.
                "linear_jerk":
                    Linear jerk vector. Requires `accelerometer` attribute enabled in
                    AgentInterface, else absent. shape=(3,). dtype=np.float32.
                "linear_velocity":
                    Vehicle velocity in x, y, and z axes. shape=(3,). dtype=np.float32.
                "pos":
                    Coordinate of the center of the vehicle bounding box's bottom plane.
                    shape=(3,). dtype=np.float64.
                "speed":
                    Vehicle speed in m/s. dtype=np.float32.
                "steering":
                    Angle of front wheels in radians [-pi, pi]. dtype=np.float32.
                "yaw_rate":
                    Rotation speed around vertical axis in rad/s [0, 2pi].
                    dtype=np.float32.
            )}

            A dictionary of event markers.
            "events": dict({
                "agents_alive_done":
                    1 if `DoneCriteria.agents_alive` is triggered, else 0.
                "collisions":
                    1 if any collisions occurred with ego vehicle, else 0.
                "not_moving":
                    1 if `DoneCriteria.not_moving` is triggered, else 0.
                "off_road":
                    1 if ego vehicle drives off road, else 0.
                "off_route":
                    1 if ego vehicle drives off mission route, else 0.
                "on_shoulder":
                    1 if ego vehicle drives on road shoulder, else 0.
                "reached_goal":
                    1 if ego vehicle reaches its goal, else 0.
                "reached_max_episode_steps":
                    1 if maximum episode steps reached, else 0.
                "wrong_way":
                    1 if ego vehicle drives in the wrong traffic direction, else 0.
            })

            Drivable area grid map. Map is binary, with 255 if a cell contains a road,
            else 0. dtype=np.uint8.
            "drivable_area_grid_map": np.ndarray

            Lidar point cloud, with the following attributes.
            "lidar_point_cloud": dict({
                "hit":
                    Binary array. 1 if an object is hit, else 0. shape(300,).
                "point_cloud":
                    Coordinates of lidar point cloud. shape=(300,3). dtype=np.float64.
                "ray_origin":
                    Ray origin coordinates. shape=(300,3). dtype=np.float64.
                "ray_vector":
                    Ray vectors. shape=(300,3). dtype=np.float64.
            })

            Mission details for the ego agent.
            "mission": dict({
                "goal_pos":
                    Achieve goal by reaching the end position. Defaults to np.array([0,0,0])
                    for no mission. shape=(3,). dtype=np.float64.
            })

            Feature array of 10 nearest neighborhood vehicles. If nearest neighbor
            vehicles are insufficient, default feature values are padded.
            "neighborhood_vehicle_states": dict({
                "box":
                    Bounding box of neighbor vehicles. Defaults to np.array([0,0,0]) per
                    vehicle. shape=(10,3). dtype=np.float32.
                "heading":
                    Heading of neighbor vehicles in radians [-pi, pi]. Defaults to
                    np.array([0]) per vehicle. shape=(10,). dtype=np.float32.
                "lane_index":
                    Lane number of neighbor vehicles. Defaults to np.array([0]) per
                    vehicle. shape=(10,). dtype=np.int8.
                "position":
                    Coordinate of the center of neighbor vehicles' bounding box's bottom
                    plane. Defaults to np.array([0,0,0]) per vehicle. shape=(10,3).
                    dtype=np.float64.
                "speed":
                    Speed of neighbor vehicles in m/s. Defaults to np.array([0]) per
                    vehicle. shape=(10,). dtype=np.float32.
            })

            Occupancy grid map. Map is binary, with 255 if a cell is occupied, else 0.
            dtype=np.uint8.
            "occupancy_grid_map": np.ndarray

            RGB image, from the top view, with ego vehicle at the center.
            shape=(height, width, 3). dtype=np.uint8.
            "top_down_rgb": np.ndarray

            Feature array of 20 waypoints ahead or in the mission route, from the
            nearest 4 lanes. If lanes or waypoints ahead are insufficient, default
            values are padded.
            "waypoint_paths": dict({
                "heading":
                    Lane heading angle at a waypoint in radians [-pi, pi]. Defaults to
                    np.array([0]) per waypoint. shape=(4,20). dtype=np.float32.
                "lane_index":
                    Lane number at a waypoint. Defaults to np.array([0]) per waypoint.
                    shape=(4,20). dtype=np.int8.
                "lane_width":
                    Lane width at a waypoint in meters. Defaults to np.array([0]) per
                    waypoint. shape=(4,20). dtype=np.float32.
                "position":
                    Coordinate of a waypoint. Defaults to np.array([0,0,0]).
                    shape=(4,20,3). dtype=np.float64.
                "speed_limit":
                    Lane speed limit at a waypoint in m/s. shape=(4,20). dtype=np.float32.
            })

            Feature array of 3 upcoming signals.  If there aren't this many signals ahead,
            default values are padded.
            "signals": dict({
                "state":
                    The state of the traffic signal.
                    See smarts.core.signal_provider.SignalLightState for interpretation.
                    Defaults to np.array([0]) per signal.  shape=(3,), dtype=np.int8.
                "stop_point":
                    The stopping point for traffic controlled by the signal, i.e., the
                    point where actors should stop when the signal is in a stop state.
                    Defaults to np.array([0, 0]) per signal.  shape=(3,2), dtype=np.float64.
                "last_changed":
                    If known, the simulation time this signal last changed its state.
                    Defaults to np.array([0]) per signal.  shape=(3,), dtype=np.float32.
            })

        })

    """

    def __init__(
        self,
        agent_interfaces: Dict[str, AgentInterface],
        observation_options: ObservationOptions,
    ) -> None:
        self._space_formats = {
            agent_id: observation_space_format(agent_interface)
            for agent_id, agent_interface in agent_interfaces.items()
        }
        self.observation_options = observation_options
        super().__init__()

    def format(self, observations: Dict[str, Observation]):
        """Formats smarts observations fixed sized containers."""
        if self.observation_options == ObservationOptions.unformatted:
            return observations
        # TODO MTA: Parallelize the conversion if possible
        active_obs = {
            agent_id: self._space_formats[agent_id].format(obs)
            for agent_id, obs in observations.items()
        }
        out_obs = active_obs
        if self.observation_options == ObservationOptions.full:
            missing_ids = set(self._space_formats.keys()) - set(active_obs.keys())
            padded_obs = {
                agent_id: space_format.space.sample()
                for agent_id, space_format in self._space_formats.items()
                if agent_id in missing_ids
            }
            for obs in padded_obs.values():
                obs["active"] = np.int64(False)
            out_obs.update(padded_obs)
        return out_obs

    @cached_property
    def space(self):
        """The observation space this should format the smarts observations to match."""
        return gym.spaces.Dict(
            {
                agent_id: space_format.space
                for agent_id, space_format in self._space_formats.items()
            }
        )
