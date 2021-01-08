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
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Optional, Union

from .controllers import ActionSpaceType
from .lidar_sensor_params import BasicLidar
from .lidar_sensor_params import SensorParams as LidarSensorParams


@dataclass
class DrivableAreaGridMap:
    """The width and height are in "pixels" and the resolution is the "size of a
    pixel". E.g. if you wanted 100m x 100m DrivableAreaGridMap but a 64x64 image representation
    you would do DrivableAreaGridMap(width=64, height=64, resolution=100/64)
    """

    width: int = 256
    height: int = 256
    resolution: float = 50 / 256


@dataclass
class OGM:
    """The width and height are in "pixels" and the resolution is the "size of a
    pixel". E.g. if you wanted 100m x 100m OGM but a 64x64 image representation
    you would do OGM(width=64, height=64, resolution=100/64)
    """

    width: int = 256
    height: int = 256
    resolution: float = 50 / 256


@dataclass
class RGB:
    """The width and height are in "pixels" and the resolution is the "size of a
    pixel". E.g. if you wanted 100m x 100m RGB but a 256x256 image representation
    you would do RGB(width=256, height=256, resolution=100/256)
    """

    width: int = 256
    height: int = 256
    resolution: float = 50 / 256


@dataclass
class Lidar:
    """Lidar point cloud observations."""

    sensor_params: LidarSensorParams = BasicLidar


@dataclass
class Waypoints:
    """The number of waypoints we want to look ahead by. The "distance" of this depends
    on the waypoint spacing set. The project default for that is one meter.
    """

    lookahead: int = 50


@dataclass
class RoadWaypoints:
    """RoadWaypoints give you waypoints along all lanes of the road an agent
    is currently on (including oncoming lanes).

    RoadWaypoint observations will be returned in a mapping from lane id to the waypoints along
    that lane.
    """

    # The distance in meters to include waypoints for (both behind and infront of the agent)
    horizon: int = 20


@dataclass
class NeighborhoodVehicles:
    # `None` means no radius filtering
    radius: float = None


@dataclass
class Accelerometer:
    pass


class AgentType(IntEnum):
    Buddha = 0
    """Agent sees nothing and does nothing"""
    Full = 1
    """All observations and continuous action space"""
    Standard = 2
    """Minimal observations for dealing with waypoints and other vehicles and
    continuous action space.
    """
    Laner = 3
    """Agent sees waypoints and performs lane actions"""
    Loner = 4
    """Agent sees waypoints and performs continuous actions"""
    Tagger = 5
    """Agent sees waypoints, other vehicles, and performs continuous actions"""
    StandardWithAbsoluteSteering = 6
    """Agent sees waypoints, neighbor vehicles and performs continuous action"""
    LanerWithSpeed = 7
    """Agent sees waypoints and performs speed and lane action"""
    Tracker = 8
    """Agent sees waypoints and performs target position action"""
    Boid = 9
    """Controls multiple vehicles"""
    MPCTracker = 10
    """Agent performs trajectory tracking using model predictive control."""


@dataclass(frozen=True)
class DoneCriteria:
    """Toggleable conditions on which to trigger episode end."""

    collision: bool = True
    """End the episode when the agent collides with another vehicle."""
    off_road: bool = True
    """End the episode when the agent drives off the road."""
    off_route: bool = True
    """End the episode when the agent drives off the specified mission route."""
    on_shoulder: bool = False
    """End the episode when the agent drives on the road shoulder."""
    wrong_way: bool = False
    """End the episode when the agent drives in the wrong direction, even though it
    may be driving on the mission route.
    """
    not_moving: bool = False
    """End the episode when the agent is not moving for 60 seconds or more. To account
    for controller noise not moving means <= 1 meter of displacement within 60 seconds.
    """


@dataclass
class AgentBehavior:
    """Agent behavior configuration."""

    aggressiveness: int = 0
    """The aggressiveness affects the waypoints of a mission task, in terms of
    the trigger timing.
    """


@dataclass
class AgentInterface:
    """
    Configure the interface between an Agent and the Environment.
    Choose the action space and sensors to enable.
    """

    debug: bool = False
    """Enable debug information for the various sensors and action spaces."""

    done_criteria: DoneCriteria = field(default_factory=lambda: DoneCriteria())
    """Configurable criteria of when to mark this actor as done. Done actors will be
    removed from the environment and may trigger the episode to be done."""

    max_episode_steps: Optional[int] = None
    """If set, agents will become "done" after this many steps. set to None to disable."""

    neighborhood_vehicles: Union[NeighborhoodVehicles, bool] = False
    """Enable the Neighborhood Vehicle States sensor, vehicles around the ego vehicle will be provided."""

    waypoints: Union[Waypoints, bool] = False
    """Enable the Waypoint Paths sensor, a list of valid waypoint paths along the current mission."""

    road_waypoints: Union[RoadWaypoints, bool] = False
    """
    Enable the Road Waypoints sensor, waypoints along all lanes (oncoming included) of the road the
    vehicle is currently on will be provided even if these waypoints do not match the current mission.
    """

    drivable_area_grid_map: Union[DrivableAreaGridMap, bool] = False
    """
    Enable the DrivableAreaGridMap sensor, a grid is provided where each cell signals whether the
    corresponding area of the map is a drivable surface.
    """

    ogm: Union[OGM, bool] = False
    """
    Enable the OGM (Occupancy Grid Map) sensor, a grid is provided where each cell signals whether
    that area in space is occupied.
    """

    rgb: Union[RGB, bool] = False
    """
    Enable the RGB camera sensor, a top down color image is provided.
    """

    lidar: Union[Lidar, bool] = False
    """
    Enable the LIDAR point cloud sensor.
    """

    action: Optional[ActionSpaceType] = None
    """
    The choice of action space, this action space also decides the controller that will be enabled.
    """

    vehicle_type: str = "sedan"
    """
    The choice of vehicle type.
    """

    accelerometer: Union[Accelerometer, bool] = True
    """
    Enable acceleration and jerk observations.
    """

    agent_behavior: AgentBehavior = None
    """
    The behavior configuration of the agent.
    """

    def __post_init__(self):
        self.neighborhood_vehicles = AgentInterface._resolve_config(
            self.neighborhood_vehicles, NeighborhoodVehicles
        )
        self.waypoints = AgentInterface._resolve_config(self.waypoints, Waypoints)
        self.road_waypoints = AgentInterface._resolve_config(
            self.road_waypoints, RoadWaypoints
        )
        self.drivable_area_grid_map = AgentInterface._resolve_config(
            self.drivable_area_grid_map, DrivableAreaGridMap
        )
        self.ogm = AgentInterface._resolve_config(self.ogm, OGM)
        self.rgb = AgentInterface._resolve_config(self.rgb, RGB)
        self.lidar = AgentInterface._resolve_config(self.lidar, Lidar)
        self.accelerometer = AgentInterface._resolve_config(
            self.accelerometer, Accelerometer
        )
        assert self.vehicle_type in {"sedan", "bus"}

    @staticmethod
    def from_type(requested_type: AgentType, **kwargs):
        """Instantiates from a selection of agent_interface presets

        Args:
            requested_type:
                Select a premade AgentInterface from an AgentType
            max_episode_steps:
                The total number of steps this interface will observe before expiring
        """
        if requested_type == AgentType.Buddha:  # The enlightened one
            interface = AgentInterface()
        elif requested_type == AgentType.Full:  # Uses everything
            interface = AgentInterface(
                neighborhood_vehicles=True,
                waypoints=True,
                drivable_area_grid_map=True,
                ogm=True,
                rgb=True,
                lidar=True,
                action=ActionSpaceType.Continuous,
            )
        # Uses low dimensional observations
        elif requested_type == AgentType.StandardWithAbsoluteSteering:
            interface = AgentInterface(
                waypoints=True,
                neighborhood_vehicles=True,
                action=ActionSpaceType.Continuous,
            )
        elif requested_type == AgentType.Standard:
            interface = AgentInterface(
                waypoints=True,
                neighborhood_vehicles=True,
                action=ActionSpaceType.ActuatorDynamic,
            )
        elif requested_type == AgentType.Laner:  # The lane-following agent
            interface = AgentInterface(waypoints=True, action=ActionSpaceType.Lane,)
        # The lane-following agent with speed and relative lane change direction
        elif requested_type == AgentType.LanerWithSpeed:
            interface = AgentInterface(
                waypoints=True, action=ActionSpaceType.LaneWithContinuousSpeed,
            )
        # The trajectory tracking agent wich recieves a series of reference trajectory
        # points and speeds to follow
        elif requested_type == AgentType.Tracker:
            interface = AgentInterface(
                waypoints=True, action=ActionSpaceType.Trajectory,
            )
        # The MPC based trajectory tracking agent wich recieves a series of
        # reference trajectory points and speeds and computes the optimal
        # steering action.
        elif requested_type == AgentType.MPCTracker:
            interface = AgentInterface(waypoints=True, action=ActionSpaceType.MPC,)
        # For boid control (controlling multiple vehicles)
        elif requested_type == AgentType.Boid:
            interface = AgentInterface(
                waypoints=True,
                neighborhood_vehicles=True,
                action=ActionSpaceType.MultiTargetPose,
            )
        # For empty environment, good for testing control
        elif requested_type == AgentType.Loner:
            interface = AgentInterface(
                waypoints=True, action=ActionSpaceType.Continuous,
            )
        # Plays tag _two vehicles in the env only_
        elif requested_type == AgentType.Tagger:
            interface = AgentInterface(
                waypoints=True, action=ActionSpaceType.Continuous,
            )
        else:
            raise Exception("Unsupported agent type %s" % requested_type)

        return interface.replace(**kwargs)

    def replace(self, **kwargs):
        """Clone this AgentInteface with the given fields updated
        >>> interface = AgentInterface(action=ActionSpaceType.Continuous) \
                            .replace(waypoints=True)
        >>> interface.waypoints
        Waypoints(...)
        """
        return replace(self, **kwargs)

    @property
    def action_space(self):
        # for backwards compatibility
        return self.action

    @staticmethod
    def _resolve_config(config, type_):
        if config is True:
            return type_()
        elif isinstance(config, type_):
            return config
        else:
            return False
