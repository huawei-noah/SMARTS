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
from typing import Dict, List

import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface
from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.sensors import Observation

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

## TODO MTA: use constants instead these classes
class _Vec3SignedFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32)


class _Vec3UnsignedFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32)


class _Vec3SignedFloat64Space:
    @property
    def space(self):
        return gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float64)


class _SignedFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=-1e10, high=1e10, dtype=np.float32)


class _UnsignedFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=0, high=1e10, dtype=np.float32)


class _SignedRadiansFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=-math.pi, high=math.pi, dtype=np.float32)


class _UnsignedRadiansFloat32Space:
    @property
    def space(self):
        return gym.spaces.Box(low=0, high=2 * math.pi, shape=(), dtype=np.float32)


class _UnsignedInt8Space:
    @property
    def space(self):
        return gym.spaces.Box(low=0, high=127, shape=(), dtype=np.int8)


class _Discrete2Space:
    @property
    def space(self):
        return gym.spaces.Discrete(n=2)


class BaseSpaceFormat:
    def format(self, obs: Observation):
        raise NotImplementedError()

    def active(self, agent_interface: AgentInterface) -> bool:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def space(self):
        raise NotImplementedError()


class HeadingSpaceFormat(_SignedRadiansFloat32Space, BaseSpaceFormat):
    @property
    def name(self):
        return "heading"

    @property
    def space(self):
        return gym.spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32)


class SpeedSpaceFormat(_UnsignedFloat32Space, BaseSpaceFormat):
    @property
    def name(self):
        return "speed"

    @property
    def space(self):
        return gym.spaces.Box(low=0, high=1e10, shape=(), dtype=np.float32)


class PositionSpaceFormat(_Vec3SignedFloat64Space, BaseSpaceFormat):
    @property
    def name(self):
        return "position"

    @property
    def space(self):
        return gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float64)


class VelocitySpaceFormat(_Vec3SignedFloat32Space, BaseSpaceFormat):
    pass


class AccelerationSpaceFormat(_Vec3SignedFloat32Space, BaseSpaceFormat):
    pass


class JerkSpaceFormat(_Vec3SignedFloat32Space, BaseSpaceFormat):
    pass


class ConfigurableSpaceFormat(BaseSpaceFormat):
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


class EgoBoxSpaceFormat(_Vec3UnsignedFloat32Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.array(obs.ego_vehicle_state.bounding_box.as_lwh).astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "box"


class EgoHeadingSpaceFormat(HeadingSpaceFormat):
    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.heading)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoLaneIndexSpaceFormat(_UnsignedInt8Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int8(obs.ego_vehicle_state.lane_index)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "lane_index"


class EgoLinearVelocitySpaceFormat(VelocitySpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_velocity.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "linear_velocity"


class EgoAngularVelocitySpaceFormat(VelocitySpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_velocity.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "angular_velocity"


class EgoPositionSpaceFormat(PositionSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.position.astype(np.float64)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoSpeedSpaceFormat(SpeedSpaceFormat):
    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.speed)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True


class EgoLaneIDSpaceFormat(BaseSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.lane_id.ljust(
            _WAYPOINT_NAME_LIMIT, _TEXT_PAD_CHAR
        )[:_WAYPOINT_NAME_LIMIT]

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "lane_id"

    @property
    def space(self):
        return gym.spaces.Text(_WAYPOINT_NAME_LIMIT, charset=_WAYPOINT_CHAR_SET)


class EgoSteeringSpaceFormat(_SignedRadiansFloat32Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.steering)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "steering"


class EgoYawRateSpaceFormat(_UnsignedRadiansFloat32Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.float32(obs.ego_vehicle_state.yaw_rate)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "yaw_rate"


class EgoAngularAccelerationSpaceFormat(AccelerationSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_acceleration.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "angular_acceleration"


class EgoAngularJerkSpaceFormat(JerkSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.angular_jerk.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "angular_jerk"


class EgoLinearAccelerationSpaceFormat(AccelerationSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_acceleration.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "linear_acceleration"


class EgoLinearJerkSpaceFormat(JerkSpaceFormat):
    def format(self, obs: Observation):
        return obs.ego_vehicle_state.linear_jerk.astype(np.float32)

    def active(self, agent_interface: AgentInterface) -> bool:
        return bool(agent_interface.accelerometer)

    @property
    def name(self):
        return "linear_jerk"


class DistanceTravelledSpace(_UnsignedFloat32Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.float32(obs.distance_travelled)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "distance_travelled"


class EgoVehicleStateSpaceFormat(DictSpaceFormat):
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


class EventsAgentsAliveDoneSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.agents_alive_done)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "agents_alive_done"


class EventsCollisionsSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(len(obs.events.collisions) > 0)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "collisions"


class EventsNotMovingSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.not_moving)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "not_moving"


class EventsOffRoadSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.off_road)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "off_road"


class EventsOffRouteSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.off_route)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "off_route"


class EventsOnShoulderSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.on_shoulder)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "on_shoulder"


class EventsReachedGoalSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.reached_goal)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "reached_goal"


class EventsReachedMaxEpisodeStepsSpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.reached_max_episode_steps)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "reached_max_episode_steps"


class EventsWrongWaySpaceFormat(_Discrete2Space, BaseSpaceFormat):
    def format(self, obs: Observation):
        return np.int64(obs.events.wrong_way)

    def active(self, agent_interface: AgentInterface) -> bool:
        return True

    @property
    def name(self):
        return "wrong_way"


class EventsSpaceFormat(DictSpaceFormat):
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


class ObservationSpaceFormat(DictSpaceFormat):
    def __init__(self, agent_interface) -> None:
        spaces = [
            EgoVehicleStateSpaceFormat(agent_interface),
            EventsSpaceFormat(agent_interface),
            DrivableAreaGridMapSpaceFormat(agent_interface),
            LidarPointCloudSpaceFormat(agent_interface),
            MissionSpaceFormat(),
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
    def __init__(self, agent_interfaces: Dict[str, AgentInterface]) -> None:
        self.space_formats = {
            agent_id: ObservationSpaceFormat(agent_interface)
            for agent_id, agent_interface in agent_interfaces.items()
        }
        super().__init__()

    def format(self, observations: Dict[str, Observation]):
        # TODO MTA: Parallelize the conversion if possible
        return {
            agent_id: self.space_formats[agent_id].format(obs)
            for agent_id, obs in observations.items()
        }

    @property
    def space(self):
        return gym.spaces.Dict(
            {
                agent_id: space_format.space
                for agent_id, space_format in self.space_formats.items()
            }
        )
