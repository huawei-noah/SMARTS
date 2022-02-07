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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.sensors import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    Observation,
    OccupancyGridMap,
    TopDownRGB,
    VehicleObservation,
)
from smarts.env.custom_observations import lane_ttc

_LIDAR_SHP = 300
_NEIGHBOR_SHP = 10
_WAYPOINT_SHP = (4, 20)


@dataclass(frozen=True)
class StdObs:
    """Observations in numpy array format, suitable for vectorized
    processing."""

    dist: np.float32
    """Total distance travelled in meters. dtype=np.float32."""
    ego: Dict[str, Union[np.int8, np.float32, np.ndarray]]
    """Ego vehicle state, with the following attributes.

    angular_acceleration:
        Angular acceleration vector. Requires `accelerometer` attribute enabled
        in AgentInterface, else defaults to array of zeros. shape=(3,). 
        dtype=np.float32.
    angular_jerk:
        Angular jerk vector. Requires `accelerometer` attribute enabled in
        AgentInterface, else defaults to array of zeros. shape=(3,). 
        dtype=np.float32.
    angular_velocity:
        Angular velocity vector. shape=(3,). dtype=np.float32).
    box:
        Length, width, and height of the vehicle bounding box. shape=(3,).
        dtype=np.float32.
    heading:
        Vehicle heading in radians [-pi, pi]. dtype=np.float32.
    lane_index:
        Vehicle's lane number. Rightmost lane has index 0 and increases towards
        left. dtype=np.int8.
    linear_acceleration:
        Vehicle acceleration in x, y, and z axes. Requires `accelerometer`
        attribute enabled in AgentInterface, else defaults to array of zeros. 
        shape=(3,). dtype=np.float32.
    linear_jerk:
        Linear jerk vector. Requires `accelerometer` attribute enabled in
        AgentInterface, else defaults to array of zeros. shape=(3,). 
        dtype=np.float32.
    linear_velocity:
        Vehicle velocity in x, y, and z axes. shape=(3,). dtype=np.float32.
    pos:
        Coordinate of the center of the vehicle bounding box's bottom plane.
        shape=(3,). dtype=np.float64.
    speed:
        Vehicle speed in m/s. dtype=np.float32.
    steering:
        Angle of front wheels in radians [-pi, pi]. dtype=np.float32.
    yaw_rate:
        Rotation speed around vertical axis in rad/s [0, 2pi]. 
        dtype=np.float32.
    """
    events: Dict[str, np.int8]
    """ A dictionary of event markers.
    
    agents_alive_done:
        1 if `DoneCriteria.agents_alive` is triggered, else 0.
    collisions:
        1 if any collisions occurred with ego vehicle, else 0.
    not_moving:
        1 if `DoneCriteria.not_moving` is triggered, else 0.
    off_road:
        1 if ego vehicle drives off road, else 0.
    off_route:
        1 if ego vehicle drives off mission route, else 0.
    on_shoulder:
        1 if ego vehicle drives on road shoulder, else 0.
    reached_goal:
        1 if ego vehicle reaches its goal, else 0.
    reached_max_episode_steps:
        1 if maximum episode steps reached, else 0.
    wrong_way:
        1 if ego vehicle drives in the wrong traffic direction, else 0.
    """
    dagm: Optional[np.ndarray] = None
    """Drivable area grid map. Map is binary, with 255 if a cell contains a
    road, else 0. dtype=np.uint8.
    """
    lidar: Optional[Dict[str, np.ndarray]] = None
    """Lidar point cloud, with the following attributes.
    
    hit:
        Binary array. 1 if an object is hit, else 0. shape(300,).
    point_cloud:
        Coordinates of lidar point cloud. shape=(300,3). dtype=np.float64.
    ray_origin:
        Ray origin coordinates. shape=(300,3). dtype=np.float64.
    ray_vector:
        Ray vectors. shape=(300,3). dtype=np.float64.
    """
    neighbors: Optional[Dict[str, np.ndarray]] = None
    """Feature array of 10 nearest neighborhood vehicles. If nearest neighbor
    vehicles are insufficient, default feature values are padded.
    
    box:
        Bounding box of neighbor vehicles. Defaults to np.array([0,0,0]) per 
        vehicle. shape=(10,3). dtype=np.float32.
    heading:
        Heading of neighbor vehicles in radians [-pi, pi]. Defaults to 
        np.array([0]) per vehicle. shape=(10,). dtype=np.float32.
    lane_index:
        Lane number of neighbor vehicles. Defaults to np.array([0]) per 
        vehicle. shape=(10,). dtype=np.int8.
    pos:
        Coordinate of the center of neighbor vehicles' bounding box's bottom 
        plane. Defaults to np.array([0,0,0]) per vehicle. shape=(10,3). 
        dtype=np.float64.
    speed:
        Speed of neighbor vehicles in m/s. Defaults to np.array([0]) per
        vehicle. shape=(10,). dtype=np.float32.
    """
    ogm: Optional[np.ndarray] = None
    """Occupancy grid map. Map is binary, with 255 if a cell is occupied, else
    0. dtype=np.uint8."""
    rgb: Optional[np.ndarray] = None
    """RGB image, from the top view, with ego vehicle at the center.
    shape=(height, width, 3). dtype=np.uint8."""
    ttc: Optional[Dict[str, Union[np.float32, np.ndarray]]] = None
    """Time and distance to collision. Enabled only if both `waypoints` and
    `neighborhood_vehicles` attributes are enabled in AgentInterface.
    
    angle_error:
        Angular error in radians [-pi, pi]. dtype=np.float32.
    distance_from_center:
        Distance of vehicle from lane center in meters. dtype=np.float32.
    dtc:
        Distance to collision on the right lane (`dtc[0]`), current lane 
        (`dtc[1]`), and left lane (`dtc[2]`). If no lane is available, to the 
        right or to the left, default value of 0 is padded. shape=(3,). 
        dtype=np.float32.
    ttc:
        Time to collision on the right lane (`ttc[0]`), current lane
        (`ttc[1]`), and left lane (`ttc[2]`). If no lane is available,
        to the right or to the left, default value of 0 is padded. shape=(3,).
        dtype=np.float32.
    """
    waypoints: Optional[Dict[str, np.ndarray]] = None
    """Feature array of 20 waypoints ahead or in the mission route, from the 
    nearest 4 lanes. If lanes or waypoints ahead are insufficient, default 
    values are padded.
    
    heading:
        Lane heading angle at a waypoint in radians [-pi, pi]. Defaults to
        np.array([0]) per waypoint. shape=(4,20). dtype=np.float32.
    lane_index:
        Lane number at a waypoint. Defaults to np.array([0]) per waypoint.
        shape=(4,20). dtype=np.int8.
    lane_width:
        Lane width at a waypoint in meters. Defaults to np.array([0]) per
        waypoint. shape=(4,20). dtype=np.float32.
    pos:
        Coordinate of a waypoint. Defaults to np.array([0,0,0]). 
        shape=(4,20,3). dtype=np.float64.
    speed_limit:
        Lane speed limit at a waypoint in m/s. shape=(4,20). dtype=np.float32.
    """


class FormatObs(gym.ObservationWrapper):
    """Converts SMARTS observations to gym-compliant vectorized observations
    and returns `StdObs`. The observation set returned depends on the features
    enabled via AgentInterface.

    Note:
        (a) FormatObs wrapper requires all agents must have the same
            AgentInterface attributes.
        (b) Observation adapters should not be used inside the `step` and
            `reset` methods of the base environment.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): SMARTS environment to be wrapped.

        Raises:
            AssertionError: If all agents do not have the same AgentInterface.
        """
        super().__init__(env)

        agent_id = next(iter(self.agent_specs.keys()))
        intrfcs = {}
        for intrfc in {
            "accelerometer",
            "drivable_area_grid_map",
            "lidar",
            "neighborhood_vehicles",
            "ogm",
            "rgb",
            "waypoints",
        }:
            val = getattr(self.agent_specs[agent_id].interface, intrfc)
            if val:
                self._cmp_intrfc(intrfc, val)
                intrfcs.update({intrfc: val})

        space = _make_space(intrfcs)
        self.observation_space = gym.spaces.Dict(
            {agent_id: gym.spaces.Dict(space) for agent_id in self.agent_specs.keys()}
        )

        self._obs = {
            "dagm": "drivable_area_grid_map",
            "dist": "distance_travelled",
            "ego": "ego_vehicle_state",
            "events": "events",
            "lidar": "lidar_point_cloud",
            "neighbors": "neighborhood_vehicle_states",
            "ogm": "occupancy_grid_map",
            "rgb": "top_down_rgb",
            "ttc": "ttc",
            "waypoints": "waypoint_paths",
        }

    def _cmp_intrfc(self, intrfc: str, val: Any):
        assert all(
            getattr(self.agent_specs[agent_id].interface, intrfc) == val
            for agent_id in self.agent_specs.keys()
        ), f"To use FormatObs wrapper, all agents must have the same "
        f"AgentInterface.{intrfc} attribute."

    def observation(self, obs: Dict[str, Any]) -> Dict[str, StdObs]:
        """Converts SMARTS observations to gym-compliant vectorized
        observations.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            wrapped_ob = {}
            for stdob, ob in self._obs.items():
                func = globals()[f"_std_{stdob}"]
                if stdob == "ttc":
                    val = func(obs[agent_id])
                else:
                    val = func(getattr(agent_obs, ob))
                wrapped_ob.update({stdob: val})
            wrapped_obs.update({agent_id: StdObs(**wrapped_ob)})

        return wrapped_obs


def intrfc_to_stdobs(intrfc: str) -> Optional[str]:
    """Returns formatted observation name corresponding to the
    AgentInterface attribute name.

    Args:
        intrfc (str): AgentInterface attribute name.

    Returns:
        Optional[str]: Corresponding formatted observation name. None, if
        unavailable.
    """
    return {
        "drivable_area_grid_map": "dagm",
        "lidar": "lidar",
        "neighborhood_vehicles": "neighbors",
        "ogm": "ogm",
        "rgb": "rgb",
        "waypoints": "waypoints",
    }.get(intrfc, None)


def get_spaces() -> Tuple[Dict[str, gym.Space], Dict[str, Callable[[Any], gym.Space]]]:
    """Returns the basic gym space and the optional gym space of a `StdObs`.

    Returns:
        Tuple[ Dict[str, gym.Space], Dict[str, Callable[[Any], gym.Space]] ]:
            Basic and optional gym space of a `StdObs`.
    """
    # fmt: off
    basic = {
        "dist": gym.spaces.Box(low=0, high=1e10, shape=(), dtype=np.float32),
        "ego": gym.spaces.Dict({
            "angular_acceleration": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "angular_jerk": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "angular_velocity": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "box": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=(), dtype=np.int8),
            "linear_acceleration": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "linear_velocity": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "linear_jerk": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float64),
            "speed": gym.spaces.Box(low=0, high=1e10, shape=(), dtype=np.float32),
            "steering": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(), dtype=np.float32),
            "yaw_rate": gym.spaces.Box(low=0, high=2*math.pi, shape=(), dtype=np.float32),
        }),
        "events": gym.spaces.Dict({
            "agents_alive_done": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "collisions": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "not_moving": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "off_road": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "off_route": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "on_shoulder": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "reached_goal": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "reached_max_episode_steps": gym.spaces.MultiDiscrete(1, dtype=np.int8),
            "wrong_way": gym.spaces.MultiDiscrete(1, dtype=np.int8),
        }),
    }

    opt = {
        "dagm": lambda val: gym.spaces.Box(low=0, high=255, shape=(val.height, val.width, 1), dtype=np.uint8),
        "lidar": lambda _: gym.spaces.Dict({
            "hit": gym.spaces.MultiBinary(_LIDAR_SHP),
            "point_cloud": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float64),
            "ray_origin": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float64),
            "ray_vector": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float64),
        }),
        "neighbors": lambda _: gym.spaces.Dict({
            "box": gym.spaces.Box(low=0, high=1e10, shape=(_NEIGHBOR_SHP,3), dtype=np.float32),
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(_NEIGHBOR_SHP,), dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=(_NEIGHBOR_SHP,), dtype=np.int8),
            "pos": gym.spaces.Box(low=-1e10, high=1e10, shape=(_NEIGHBOR_SHP,3), dtype=np.float64),    
            "speed": gym.spaces.Box(low=0, high=1e10, shape=(_NEIGHBOR_SHP,), dtype=np.float32),
        }),
        "ogm": lambda val: gym.spaces.Box(low=0, high=255,shape=(val.height, val.width, 1), dtype=np.uint8),
        "rgb": lambda val: gym.spaces.Box(low=0, high=255, shape=(val.height, val.width, 3), dtype=np.uint8),
        "ttc": lambda _: gym.spaces.Dict({
            "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
            "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(), dtype=np.float32),
            "dtc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "ttc": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
        }),
        "waypoints": lambda _: gym.spaces.Dict({
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=_WAYPOINT_SHP, dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=_WAYPOINT_SHP, dtype=np.int8),
            "lane_width": gym.spaces.Box(low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32),
            "pos": gym.spaces.Box(low=-1e10, high=1e10, shape=_WAYPOINT_SHP + (3,), dtype=np.float64),
            "speed_limit": gym.spaces.Box(low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32),
        }),
    }
    # fmt: on

    return basic, opt


def _make_space(intrfcs: Dict[str, Any]) -> gym.Space:
    space, opt_space = get_spaces()

    for intrfc, val in intrfcs.items():
        opt_ob = intrfc_to_stdobs(intrfc)
        if opt_ob:
            space.update({opt_ob: opt_space[opt_ob](val)})

    if "waypoints" in intrfcs.keys() and "neighborhood_vehicles" in intrfcs.keys():
        space.update({"ttc": opt_space["ttc"](None)})

    return space


def _std_dagm(
    val: Optional[DrivableAreaGridMap],
) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_dist(val: float) -> float:
    return np.float32(val)


def _std_ego(
    val: EgoVehicleObservation,
) -> Dict[str, Union[np.int8, np.float32, np.ndarray]]:

    if val.angular_acceleration is None:
        ang_accel = np.zeros((3,), dtype=np.float32)
        ang_jerk = np.zeros((3,), dtype=np.float32)
        lin_accel = np.zeros((3,), dtype=np.float32)
        lin_jerk = np.zeros((3,), dtype=np.float32)
    else:
        ang_accel = val.angular_acceleration.astype(np.float32)
        ang_jerk = val.angular_jerk.astype(np.float32)
        lin_accel = val.linear_acceleration.astype(np.float32)
        lin_jerk = val.linear_jerk.astype(np.float32)

    return {
        "angular_acceleration": ang_accel,
        "angular_jerk": ang_jerk,
        "angular_velocity": val.angular_velocity.astype(np.float32),
        "box": np.array(val.bounding_box.as_lwh).astype(np.float32),
        "heading": np.float32(val.heading),
        "lane_index": np.int8(val.lane_index),
        "linear_acceleration": lin_accel,
        "linear_jerk": lin_jerk,
        "linear_velocity": val.linear_velocity.astype(np.float32),
        "pos": val.position.astype(np.float64),
        "speed": np.float32(val.speed),
        "steering": np.float32(val.steering),
        "yaw_rate": np.float32(val.yaw_rate),
    }


def _std_events(val: Events) -> Dict[str, int]:
    return {
        "agents_alive_done": np.int8(val.agents_alive_done),
        "collisions": np.int8(len(val.collisions) > 0),
        "not_moving": np.int8(val.not_moving),
        "off_road": np.int8(val.off_road),
        "off_route": np.int8(val.off_route),
        "on_shoulder": np.int8(val.on_shoulder),
        "reached_goal": np.int8(val.reached_goal),
        "reached_max_episode_steps": np.int8(val.reached_max_episode_steps),
        "wrong_way": np.int8(val.wrong_way),
    }


def _std_lidar(
    val: Optional[
        Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
    ]
) -> Optional[Dict[str, np.ndarray]]:
    if not val:
        return None

    des_shp = _LIDAR_SHP
    hit = np.array(val[1], dtype=np.int8)
    point_cloud = np.array(val[0], dtype=np.float64)
    point_cloud = np.nan_to_num(
        point_cloud,
        copy=False,
        nan=np.float64(0),
        posinf=np.float64(0),
        neginf=np.float64(0),
    )
    ray_origin, ray_vector = zip(*(val[2]))
    ray_origin = np.array(ray_origin, np.float64)
    ray_vector = np.array(ray_vector, np.float64)

    try:
        assert hit.shape == (des_shp,)
        assert point_cloud.shape == (des_shp, 3)
        assert ray_origin.shape == (des_shp, 3)
        assert ray_vector.shape == (des_shp, 3)
    except:
        raise Exception("Internal Error: Mismatched lidar point cloud shape.")

    return {
        "hit": hit,
        "point_cloud": point_cloud,
        "ray_origin": ray_origin,
        "ray_vector": ray_vector,
    }


def _std_neighbors(
    nghbs: Optional[List[VehicleObservation]],
) -> Optional[Dict[str, np.ndarray]]:
    if not nghbs:
        return None

    des_shp = _NEIGHBOR_SHP
    rcv_shp = len(nghbs)
    pad_shp = 0 if des_shp - rcv_shp < 0 else des_shp - rcv_shp

    nghbs = [
        (
            nghb.bounding_box.as_lwh,
            nghb.heading,
            nghb.lane_index,
            nghb.position,
            nghb.speed,
        )
        for nghb in nghbs[:des_shp]
    ]
    box, heading, lane_index, pos, speed = zip(*nghbs)

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
        "box": box,
        "heading": heading,
        "lane_index": lane_index,
        "pos": pos,
        "speed": speed,
    }


def _std_ogm(val: Optional[OccupancyGridMap]) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_rgb(val: Optional[TopDownRGB]) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_ttc(obs: Observation) -> Optional[Dict[str, Union[np.float32, np.ndarray]]]:
    if not obs.neighborhood_vehicle_states or not obs.waypoint_paths:
        return None

    val = lane_ttc(obs)
    return {
        "angle_error": np.float32(val["angle_error"][0]),
        "distance_from_center": np.float32(val["distance_from_center"][0]),
        "dtc": np.array(val["ego_lane_dist"], dtype=np.float32),
        "ttc": np.array(val["ego_ttc"], dtype=np.float32),
    }


def _std_waypoints(
    paths: Optional[List[List[Waypoint]]],
) -> Optional[Dict[str, np.ndarray]]:
    if not paths:
        return None

    des_shp = _WAYPOINT_SHP
    rcv_shp = (len(paths), len(paths[0]))
    pad_shp = [0 if des - rcv < 0 else des - rcv for des, rcv in zip(des_shp, rcv_shp)]

    def extract_elem(waypoint):
        return (
            waypoint.heading,
            waypoint.lane_index,
            waypoint.lane_width,
            waypoint.pos,
            waypoint.speed_limit,
        )

    paths = [map(extract_elem, path[: des_shp[1]]) for path in paths[: des_shp[0]]]
    heading, lane_index, lane_width, pos, speed_limit = zip(
        *[zip(*path) for path in paths]
    )

    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    lane_width = np.array(lane_width, dtype=np.float32)
    pos = np.array(pos, dtype=np.float64)
    speed_limit = np.array(speed_limit, dtype=np.float32)

    # fmt: off
    heading = np.pad(heading, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_width = np.pad(lane_width, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp[0]),(0,pad_shp[1]),(0,1)), mode='constant', constant_values=0)
    speed_limit = np.pad(speed_limit, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    # fmt: on

    return {
        "heading": heading,
        "lane_index": lane_index,
        "lane_width": lane_width,
        "pos": pos,
        "speed_limit": speed_limit,
    }
