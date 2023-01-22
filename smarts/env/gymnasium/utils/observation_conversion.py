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
from typing import Dict, List

import gymnasium as gym
import numpy as np
from cached_property import cached_property

from smarts.core.agent_interface import AgentInterface
from smarts.core.events import Events
from smarts.core.observations import Observation
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
_UNSIGNED_FLOAT32_SPACE = gym.spaces.Box(low=0, high=1e10, shape=(), dtype=np.float32)
_SIGNED_RADIANS_FLOAT32_SPACE = gym.spaces.Box(
    low=-math.pi, high=math.pi, shape=(), dtype=np.float32
)
_UNSIGNED_RADIANS_FLOAT32_SPACE = gym.spaces.Box(
    low=0, high=2 * math.pi, shape=(), dtype=np.float32
)
_UNSIGNED_INT8_SPACE = gym.spaces.Box(low=0, high=127, shape=(), dtype=np.int8)
_DISCRETE2_SPACE = gym.spaces.Discrete(n=2)


class ObservationOptions(IntEnum):
    """Defines the options for how the formatting matches the observation space."""

    multi_agent = 0
    """Observation partially matches observation space. Only active agents are included."""
    full = 1
    """Observation fully matches observation space. Inactive and active agents are included."""
    default = 0
    """Defaults to :attr:`multi_agent`."""


class BaseSpaceFormat:
    """Defines the base interface for an observation formatter."""

    def format(self, obs: Observation):
        """Selects and formats the given observation to get a value that matches :attr:`space`."""
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


class HeadingSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to heading in radians."""

    @property
    def name(self):
        return "heading"

    @property
    def space(self):
        return _SIGNED_RADIANS_FLOAT32_SPACE


class SpeedSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to a speed scalar."""

    @property
    def name(self):
        return "speed"

    @property
    def space(self):
        return _UNSIGNED_FLOAT32_SPACE


class PositionSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to physical position."""

    @property
    def name(self):
        return "position"

    @property
    def space(self):
        return _VEC3_SIGNED_FLOAT64_SPACE


class VelocitySpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to physical velocity."""

    @property
    def space(self):
        return _VEC3_SIGNED_FLOAT32_SPACE


class AccelerationSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to physical acceleration."""

    @property
    def space(self):
        return _VEC3_SIGNED_FLOAT32_SPACE


class JerkSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which relates to physical jerk."""

    @property
    def space(self):
        return _VEC3_SIGNED_FLOAT32_SPACE


class ConfigurableSpaceFormat(BaseSpaceFormat):
    """Defines a formatter which is dynamic."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        self._agent_interface = agent_interface

    def active(self, agent_interface: AgentInterface) -> bool:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def space(self):
        raise NotImplementedError()


class Image8BSpaceFormat(ConfigurableSpaceFormat):
    """Defines a observation formatter which is an 8-bit image."""

    def __init__(
        self, agent_interface: AgentInterface, dimensions, colors: int
    ) -> None:
        self._dimensions = dimensions
        self._colors = colors
        super().__init__(agent_interface)

    @property
    def space(self):
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._dimensions.height, self._dimensions.width, self._colors),
            dtype=np.uint8,
        )


class DictSpaceFormat(ConfigurableSpaceFormat):
    """Defines a observation formatter that contains other formatters."""

    def __init__(self, agent_interface, spaces: List[BaseSpaceFormat]) -> None:
        self._spaces = [space for space in spaces if space.active(agent_interface)]
        super().__init__(agent_interface)

    def format(self, obs: Observation):
        return {s.name: s.format(obs) for s in self._spaces}

    def active(self, agent_interface: AgentInterface) -> bool:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def space(self):
        return gym.spaces.Dict({s.name: s.space for s in self._spaces})


class EgoBoxSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.bounding_box`."""

    def format(self, obs: Observation):
        return np.array(obs.ego_vehicle_state.bounding_box.as_lwh).astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "box"

    @property
    def space(self):
        return _VEC3_UNSIGNED_FLOAT32_SPACE


class EgoHeadingSpaceFormat(HeadingSpaceFormat):
    """Formats for `obs.ego_vehicle_state.heading`."""

    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.heading)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoLaneIndexSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.lane_index`."""

    def format(self, obs: Observation):
        return np.int8(obs.ego_vehicle_state.lane_index)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "lane_index"

    @property
    def space(self):
        return _UNSIGNED_INT8_SPACE


class EgoLinearVelocitySpaceFormat(VelocitySpaceFormat):
    """Formats for `obs.ego_vehicle_state.linear_velocity`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_velocity.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "linear_velocity"


class EgoAngularVelocitySpaceFormat(VelocitySpaceFormat):
    """Formats for `obs.ego_vehicle_state.angular_velocity`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_velocity.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "angular_velocity"


class EgoPositionSpaceFormat(PositionSpaceFormat):
    """Formats for `obs.ego_vehicle_state.position`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.position.astype(np.float64)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoSpeedSpaceFormat(SpeedSpaceFormat):
    """Formats for `obs.ego_vehicle_state.speed`."""

    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.speed)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoLaneIDSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.lane_id`."""

    def format(self, obs: Observation):
        lane_name = obs.ego_vehicle_state.lane_id.ljust(
            _WAYPOINT_NAME_LIMIT, _TEXT_PAD_CHAR
        )
        if len(lane_name) > _WAYPOINT_NAME_LIMIT:
            warnings.warn(
                f"Lane named `{lane_name}` is more than "
                f"`{_WAYPOINT_NAME_LIMIT}` characters long. It will be truncated "
                "and may cause unintended issues with navigation and lane identification."
            )
        return lane_name[:_WAYPOINT_NAME_LIMIT]

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "lane_id"

    @property
    def space(self):
        return gym.spaces.Text(_WAYPOINT_NAME_LIMIT, charset=_WAYPOINT_CHAR_SET)


class EgoSteeringSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.steering`."""

    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.steering)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "steering"

    @property
    def space(self):
        return _SIGNED_RADIANS_FLOAT32_SPACE


class EgoYawRateSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.yaw_rate`."""

    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.yaw_rate)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "yaw_rate"

    @property
    def space(self):
        return _UNSIGNED_RADIANS_FLOAT32_SPACE


class EgoAngularAccelerationSpaceFormat(AccelerationSpaceFormat):
    """Formats for `obs.ego_vehicle_state.angular_acceleration`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_acceleration.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "angular_acceleration"


class EgoAngularJerkSpaceFormat(JerkSpaceFormat):
    """Formats for `obs.ego_vehicle_state.angular_jerk`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_jerk.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "angular_jerk"


class EgoLinearAccelerationSpaceFormat(AccelerationSpaceFormat):
    """Formats for `obs.ego_vehicle_state.linear_acceleration`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_acceleration.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "linear_acceleration"


class EgoLinearJerkSpaceFormat(JerkSpaceFormat):
    """Formats for `obs.ego_vehicle_state.linear_jerk`."""

    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_jerk.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "linear_jerk"


class DistanceTravelledSpace(BaseSpaceFormat):
    """Formats for `obs.distance_travelled`."""

    def format(self, obs: Observation):
        return np.float32(obs.distance_travelled)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "distance_travelled"

    @property
    def space(self):
        return _UNSIGNED_FLOAT32_SPACE


class EgoVehicleStateSpaceFormat(DictSpaceFormat):
    """Formats for `obs.ego_vehicle_state`."""

    def __init__(self, agent_interface) -> None:
        spaces = [
            # required
            EgoAngularVelocitySpaceFormat(),
            EgoBoxSpaceFormat(),
            EgoHeadingSpaceFormat(),
            EgoLaneIDSpaceFormat(),
            EgoLaneIndexSpaceFormat(),
            EgoLinearVelocitySpaceFormat(),
            EgoPositionSpaceFormat(),
            EgoSpeedSpaceFormat(),
            EgoSteeringSpaceFormat(),
            EgoYawRateSpaceFormat(),
            MissionSpaceFormat(),
            # optional
            EgoAngularAccelerationSpaceFormat(),
            EgoAngularJerkSpaceFormat(),
            EgoLinearAccelerationSpaceFormat(),
            EgoLinearJerkSpaceFormat(),
        ]

        super().__init__(agent_interface, spaces)

    def active(self, agent_interface: AgentInterface):
        return True

    @property
    def name(self):
        return "ego_vehicle_state"


class EventsAgentsAliveDoneSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.agents_alive_done`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.agents_alive_done)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "agents_alive_done"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsCollisionsSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.collisions`."""

    def format(self, obs: Observation):
        return np.int64(len(obs.events.collisions) > 0)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "collisions"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsNotMovingSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.not_moving`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.not_moving)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "not_moving"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsOffRoadSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.off_road`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.off_road)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "off_road"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsOffRouteSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.off_route`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.off_route)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "off_route"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsOnShoulderSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.on_shoulder`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.on_shoulder)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "on_shoulder"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsReachedGoalSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.reached_goal`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.reached_goal)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "reached_goal"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsReachedMaxEpisodeStepsSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.reached_max_episode_steps`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.reached_max_episode_steps)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "reached_max_episode_steps"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsWrongWaySpaceFormat(BaseSpaceFormat):
    """Formats for `obs.events.wrong_way`."""

    def format(self, obs: Observation):
        return np.int64(obs.events.wrong_way)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "wrong_way"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class EventsSpaceFormat(DictSpaceFormat):
    """Formats for `obs.events`."""

    def __init__(self, agent_interface) -> None:
        spaces = [
            EventsAgentsAliveDoneSpaceFormat(),
            EventsCollisionsSpaceFormat(),
            EventsNotMovingSpaceFormat(),
            EventsOffRoadSpaceFormat(),
            EventsOffRouteSpaceFormat(),
            EventsOnShoulderSpaceFormat(),
            EventsReachedGoalSpaceFormat(),
            EventsReachedMaxEpisodeStepsSpaceFormat(),
            EventsWrongWaySpaceFormat(),
        ]

        super().__init__(agent_interface, spaces)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "events"


class DrivableAreaGridMapSpaceFormat(Image8BSpaceFormat):
    """Formats for `obs.drivable_area_grid_map`."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        super().__init__(
            agent_interface, dimensions=agent_interface.drivable_area_grid_map, colors=1
        )

    def format(self, obs: Observation):
        return obs.drivable_area_grid_map.data.astype(np.uint8)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.drivable_area_grid_map)

    @property
    def name(self):
        return "drivable_area_grid_map"


class LidarPointCloudSpaceFormat(ConfigurableSpaceFormat):
    """Formats for `obs.lidar_point_cloud`."""

    def format(self, obs: Observation):
        lidar_point_cloud = obs.lidar_point_cloud
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
            raise Exception(
                "Internal Error: Mismatched lidar point cloud shape."
            ) from exc

        return {
            "hit": hit,
            "point_cloud": point_cloud,
            "ray_origin": ray_origin,
            "ray_vector": ray_vector,
        }

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.lidar_point_cloud)

    @property
    def name(self):
        return "lidar_point_cloud"

    @property
    def space(self):
        # MTA TODO: add lidar configuration
        return gym.spaces.Dict(
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
        )


class MissionSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.ego_vehicle_state.mission`."""

    def format(self, obs: Observation):
        ego_vehicle_obs = obs.ego_vehicle_state
        if hasattr(ego_vehicle_obs.mission.goal, "position"):
            goal_pos = np.array(ego_vehicle_obs.mission.goal.position, dtype=np.float64)
        else:
            goal_pos = np.zeros((3,), dtype=np.float64)

        return {"goal_position": goal_pos}

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "mission"

    @property
    def space(self):
        return gym.spaces.Dict(
            {
                "goal_position": gym.spaces.Box(
                    low=-1e10, high=1e10, shape=_POSITION_SHP, dtype=np.float64
                )
            }
        )


class NeighborhoodVehicleStatesSpaceFormat(ConfigurableSpaceFormat):
    """Formats for `obs.neighborhood_vehicle_states`."""

    def format(self, obs: Observation):
        neighborhood_vehicle_states = obs.neighborhood_vehicle_states
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

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.neighborhood_vehicle_states)

    @property
    def name(self):
        return "neighborhood_vehicle_states"

    @property
    def space(self):
        return gym.spaces.Dict(
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
        )


class OccupancyGridMapSpaceFormat(Image8BSpaceFormat):
    """Formats for `obs.occupancy_grid_map`."""

    def __init__(self, agent_interface: AgentInterface) -> None:
        super().__init__(
            agent_interface, dimensions=agent_interface.occupancy_grid_map, colors=1
        )

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
        super().__init__(
            agent_interface, dimensions=agent_interface.top_down_rgb, colors=3
        )

    def format(self, obs: Observation):
        return obs.top_down_rgb.data.astype(np.uint8)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.top_down_rgb)

    @property
    def name(self):
        return "top_down_rgb"


class WaypointPathsSpaceFormat(BaseSpaceFormat):
    """Formats for `obs.waypoint_paths`."""

    @property
    def name(self):
        return "waypoint_paths"

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.waypoint_paths)

    def format(self, obs: Observation):
        waypoint_paths = obs.waypoint_paths

        # Truncate all paths to be of the same length
        min_len = min(map(len, waypoint_paths))
        trunc_paths = list(map(lambda x: x[:min_len], waypoint_paths))

        des_shp = _WAYPOINT_SHP
        rcv_shp = (len(trunc_paths), len(trunc_paths[0]))
        pad_shp = [
            0 if des - rcv < 0 else des - rcv for des, rcv in zip(des_shp, rcv_shp)
        ]

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

    @property
    def space(self):
        return gym.spaces.Dict(
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
        )


class SignalsSpaceFormat(ConfigurableSpaceFormat):
    """Formats for `obs.signals`."""

    def __init__(self, agent_interface, count) -> None:
        self.count = count
        super().__init__(agent_interface)

    def format(self, obs: Observation):
        signals = obs.signals
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

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.signals)

    @property
    def name(self):
        return "signals"

    @property
    def space(self):
        return gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=0, high=32, shape=_SIGNALS_SHP, dtype=np.int8
                ),
                "stop_point": gym.spaces.Box(
                    low=-1e10, high=1e10, shape=_SIGNALS_SHP + (2,), dtype=np.float64
                ),
                "last_changed": gym.spaces.Box(
                    low=0, high=1e10, shape=_SIGNALS_SHP, dtype=np.float32
                ),
            }
        )


class EnabledSpaceFormat(BaseSpaceFormat):
    """Formats to show if the agent is active in the environment."""

    def format(self, obs: Observation):
        return np.int64(True)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "active"

    @property
    def space(self):
        return _DISCRETE2_SPACE


class ObservationSpaceFormat(DictSpaceFormat):
    """Formats a smarts observation to fixed sized object."""

    def __init__(self, agent_interface) -> None:
        spaces = [
            EnabledSpaceFormat(),
            EgoVehicleStateSpaceFormat(agent_interface),
            EventsSpaceFormat(agent_interface),
            DrivableAreaGridMapSpaceFormat(agent_interface),
            LidarPointCloudSpaceFormat(agent_interface),
            NeighborhoodVehicleStatesSpaceFormat(agent_interface),
            OccupancyGridMapSpaceFormat(agent_interface),
            TopDownRGBSpaceFormat(agent_interface),
            WaypointPathsSpaceFormat(),
            SignalsSpaceFormat(agent_interface, 3),
        ]

        super().__init__(agent_interface, spaces)

    def active(self, agent_interface: AgentInterface):
        return True

    @property
    def name(self):
        return "observation"


class ObservationsSpaceFormat:
    """Formats a smarts observation to fixed sized object.
    Observations in numpy array format, suitable for vectorized processing.

    For each agent id:
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
            "pos":
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
            "pos":
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
            agent_id: ObservationSpaceFormat(agent_interface)
            for agent_id, agent_interface in agent_interfaces.items()
        }
        self.observation_options = observation_options
        super().__init__()

    def format(self, observations: Dict[str, Observation]):
        """Formats smarts observations fixed sized containers."""
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
