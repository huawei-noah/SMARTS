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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import warnings
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import List, Optional, Union

from smarts.core.controllers import ActionSpaceType
from smarts.core.lidar_sensor_params import BasicLidar
from smarts.core.lidar_sensor_params import SensorParams as LidarSensorParams


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

    lookahead: int = 32


@dataclass
class RoadWaypoints:
    """RoadWaypoints give you waypoints along all lanes of the road an agent
    is currently on (including oncoming lanes).

    RoadWaypoint observations will be returned in a mapping from lane id to the waypoints along
    that lane.
    """

    # The distance in meters to include waypoints for (both behind and in front of the agent)
    horizon: int = 20


@dataclass
class NeighborhoodVehicles:
    """Detection of nearby vehicles and configuration for filtering of the vehicles."""

    radius: Optional[float] = None
    """The distance within which neighborhood vehicles are detected. `None` means vehicles will be detected within an unlimited distance."""


@dataclass
class Accelerometer:
    """Requires detection of motion changes within the agents vehicle."""

    pass


@dataclass
class LanePositions:
    """Computation and reporting of lane-relative RefLine (Frenet) coordinates for all vehicles."""

    pass


@dataclass
class Signals:
    """Reporting of traffic signals (lights) state in the lanes ahead."""

    lookahead: float = 100.0
    """The distance in meters to look ahead of the vehicle's current position."""


class AgentType(IntEnum):
    """Used to select preconfigured agent interfaces."""

    Buddha = 0
    """Agent sees nothing and does nothing."""
    Full = 1
    """All observations and continuous action space."""
    Standard = 2
    """Minimal observations for dealing with waypoints and other vehicles and
    continuous action space.
    """
    Laner = 3
    """Agent sees waypoints and performs lane actions."""
    Loner = 4
    """Agent sees waypoints and performs continuous actions."""
    Tagger = 5
    """Agent sees waypoints, other vehicles, and performs continuous actions."""
    StandardWithAbsoluteSteering = 6
    """Agent sees waypoints, neighbor vehicles and performs continuous action."""
    LanerWithSpeed = 7
    """Agent sees waypoints and performs speed and lane action."""
    Tracker = 8
    """Agent sees waypoints and performs target position action."""
    Boid = 9
    """Controls multiple vehicles."""
    MPCTracker = 10
    """Agent performs trajectory tracking using model predictive control."""
    TrajectoryInterpolator = 11
    """Agent performs linear trajectory interpolation."""
    Direct = 12
    """Agent sees neighbor vehicles and performs actions based on direct updates to (acceleration, angular_velocity)."""


@dataclass(frozen=True)
class AgentsListAlive:
    """Describes agents that are active in the simulation."""

    agents_list: List[str]
    """The list of agents to check whether they are alive"""
    minimum_agents_alive_in_list: int
    """Triggers the agent to be done if the number of alive agents in agents_list falls below the given value"""


@dataclass(frozen=True)
class AgentsAliveDoneCriteria:
    """Multi-agent requirements used to determine if an agent should be removed from an episode."""

    minimum_ego_agents_alive: Optional[int] = None
    """If set, triggers the agent to be done if the total number of alive ego agents falls below the given value."""
    minimum_total_agents_alive: Optional[int] = None
    """If set, triggers the agent to be done if total number of alive agents falls below the given value."""
    agent_lists_alive: Optional[List[AgentsListAlive]] = None
    """A termination criteria based on the ids of agents. If set, triggers the agent to be done if any list of agents fails 
    to meet its specified minimum number of alive agents.
    Example: 
    
    .. code-block:: python

        [
        AgentsListAlive(
            agents_list=['agent1','agent2'], minimum_agents_alive_in_list=1
        ),
        AgentsListAlive(
            agents_list=['agent3'], minimum_agents_alive_in_list=1
        ),
        ]
        
    This agent's done event would be triggered if both 'agent1' and 'agent2' is done *or* 'agent3' is done.
    """


@dataclass(frozen=True)
class EventConfiguration:
    """Configure the conditions in which an Event is triggered."""

    not_moving_time: float = 60
    """How long in seconds to check the agent after which it is considered not moving"""
    not_moving_distance: float = 1
    """How many meters the agent's actor has to move under which the agent is considered not moving"""


@dataclass(frozen=True)
class DoneCriteria:
    """Toggleable conditions on which cause removal of an agent from the current episode."""

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
    """End the episode when the agent is not moving for n seconds or more. To account
    for controller noise not moving means <= 1 meter of displacement within n seconds.
    """
    agents_alive: Optional[AgentsAliveDoneCriteria] = None
    """If set, triggers the ego agent to be done based on the number of active agents for multi-agent purposes."""


@dataclass
class AgentInterface:
    """
    Configure the interface between an Agent and the Environment.
    Choose the action space and sensors to enable.
    """

    debug: bool = False
    """Enable debug information for the various sensors and action spaces."""

    event_configuration: EventConfiguration = EventConfiguration()
    """Configurable criteria of when to trigger events"""

    done_criteria: DoneCriteria = field(default_factory=lambda: DoneCriteria())
    """Configurable criteria of when to mark this actor as done. Done actors will be
    removed from the environment and may trigger the episode to be done."""

    max_episode_steps: Optional[int] = None
    """If set, agents will become "done" after this many steps. set to None to disable."""

    neighborhood_vehicle_states: Union[NeighborhoodVehicles, bool] = False
    """Enable the Neighborhood Vehicle States sensor, vehicles around the ego vehicle will be provided."""

    waypoint_paths: Union[Waypoints, bool] = False
    """Enable the Waypoint Paths sensor, a list of valid waypoint paths along the current mission."""

    # XXX: consider making this return LanePoints instead?
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

    occupancy_grid_map: Union[OGM, bool] = False
    """
    Enable the OGM (Occupancy Grid Map) sensor, a grid is provided where each cell signals whether
    that area in space is occupied.
    """

    top_down_rgb: Union[RGB, bool] = False
    """
    Enable the RGB camera sensor, a top down color image is provided.
    """

    lidar_point_cloud: Union[Lidar, bool] = False
    """
    Enable the LIDAR point cloud sensor.
    """

    action: Optional[ActionSpaceType] = None
    """
    The choice of action space; this also decides the controller that will be enabled and the chassis type that will be used.
    """

    vehicle_type: str = "sedan"
    """
    The choice of vehicle type.
    """

    accelerometer: Union[Accelerometer, bool] = True
    """
    Enable acceleration and jerk observations.
    """

    lane_positions: Union[LanePositions, bool] = True
    """
    Enable lane-relative position reporting.
    """

    signals: Union[Signals, bool] = False
    """
    Enable the signals sensor.
    """

    def __post_init__(self):
        self.neighborhood_vehicle_states = AgentInterface._resolve_config(
            self.neighborhood_vehicle_states, NeighborhoodVehicles
        )
        self.waypoint_paths = AgentInterface._resolve_config(
            self.waypoint_paths, Waypoints
        )
        self.road_waypoints = AgentInterface._resolve_config(
            self.road_waypoints, RoadWaypoints
        )
        self.drivable_area_grid_map = AgentInterface._resolve_config(
            self.drivable_area_grid_map, DrivableAreaGridMap
        )
        self.occupancy_grid_map = AgentInterface._resolve_config(
            self.occupancy_grid_map, OGM
        )
        self.top_down_rgb = AgentInterface._resolve_config(self.top_down_rgb, RGB)
        self.lidar_point_cloud = AgentInterface._resolve_config(
            self.lidar_point_cloud, Lidar
        )
        self.accelerometer = AgentInterface._resolve_config(
            self.accelerometer, Accelerometer
        )
        self.lane_positions = AgentInterface._resolve_config(
            self.lane_positions, LanePositions
        )
        self.signals = AgentInterface._resolve_config(self.signals, Signals)
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
            interface = AgentInterface(action=ActionSpaceType.Empty)
        elif requested_type == AgentType.Full:  # Uses everything
            interface = AgentInterface(
                neighborhood_vehicle_states=True,
                waypoint_paths=True,
                drivable_area_grid_map=True,
                occupancy_grid_map=True,
                top_down_rgb=True,
                lidar_point_cloud=True,
                signals=True,
                action=ActionSpaceType.Continuous,
            )
        # Uses low dimensional observations
        elif requested_type == AgentType.StandardWithAbsoluteSteering:
            interface = AgentInterface(
                waypoint_paths=True,
                neighborhood_vehicle_states=True,
                action=ActionSpaceType.Continuous,
            )
        elif requested_type == AgentType.Standard:
            interface = AgentInterface(
                waypoint_paths=True,
                neighborhood_vehicle_states=True,
                action=ActionSpaceType.ActuatorDynamic,
            )
        elif requested_type == AgentType.Laner:  # The lane-following agent
            interface = AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.Lane,
            )
        # The lane-following agent with speed and relative lane change direction
        elif requested_type == AgentType.LanerWithSpeed:
            interface = AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.LaneWithContinuousSpeed,
            )
        # The trajectory tracking agent which receives a series of reference trajectory
        # points and speeds to follow
        elif requested_type == AgentType.Tracker:
            interface = AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.Trajectory,
            )
        # The trajectory interpolation agent which receives a with-time-trajectory and move vehicle
        # with linear time interpolation
        elif requested_type == AgentType.TrajectoryInterpolator:
            interface = AgentInterface(action=ActionSpaceType.TrajectoryWithTime)
        # The MPC based trajectory tracking agent which receives a series of
        # reference trajectory points and speeds and computes the optimal
        # steering action.
        elif requested_type == AgentType.MPCTracker:
            interface = AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.MPC,
            )
        # For boid control (controlling multiple vehicles)
        elif requested_type == AgentType.Boid:
            interface = AgentInterface(
                waypoint_paths=True,
                neighborhood_vehicle_states=True,
                action=ActionSpaceType.MultiTargetPose,
            )
        # For empty environment, good for testing control
        elif requested_type == AgentType.Loner:
            interface = AgentInterface(
                waypoint_paths=True,
                action=ActionSpaceType.Continuous,
            )
        # Plays tag _two vehicles in the env only_
        elif requested_type == AgentType.Tagger:
            interface = AgentInterface(
                waypoint_paths=True,
                neighborhood_vehicle_states=True,
                action=ActionSpaceType.Continuous,
            )
        # Has been useful for testing imitation learners
        elif requested_type == AgentType.Direct:
            interface = AgentInterface(
                neighborhood_vehicle_states=True,
                signals=True,
                action=ActionSpaceType.Direct,
            )
        else:
            raise Exception("Unsupported agent type %s" % requested_type)

        return interface.replace(**kwargs)

    def replace(self, **kwargs):
        """Clone this AgentInterface with the given fields updated
        >>> interface = AgentInterface(action=ActionSpaceType.Continuous) \
                            .replace(waypoint_paths=True)
        >>> interface.waypoint_paths
        Waypoints(...)
        """
        return replace(self, **kwargs)

    @property
    def ogm(self):
        """Deprecated. Use `occupancy_grid_map` instead."""
        warnings.warn(
            "ogm property has been deprecated in favor of occupancy_grid_map.  Please update your code.",
            category=DeprecationWarning,
        )
        return self.occupancy_grid_map

    @property
    def rgb(self):
        """Deprecated. Use `top_down_rgb` instead."""
        warnings.warn(
            "rgb property has been deprecated in favor of top_down_rgb.  Please update your code.",
            category=DeprecationWarning,
        )
        # for backwards compatibility
        return self.top_down_rgb

    @property
    def lidar(self):
        """Deprecated. Use `lidar_point_cloud` instead."""
        warnings.warn(
            "lidar property has been deprecated in favor of lidar_point_cloud.  Please update your code.",
            category=DeprecationWarning,
        )
        # for backwards compatibility
        return self.lidar_point_cloud

    @property
    def waypoints(self):
        """Deprecated. Use `waypoint_paths` instead."""
        warnings.warn(
            "waypoints property has been deprecated in favor of waypoint_paths.  Please update your code.",
            category=DeprecationWarning,
        )
        # for backwards compatibility
        return self.waypoint_paths

    @property
    def neighborhood_vehicles(self):
        """Deprecated. Use `neighborhood_vehicle_states` instead."""
        warnings.warn(
            "neighborhood_vehicles property has been deprecated in favor of neighborhood_vehicle_states.  Please update your code.",
            category=DeprecationWarning,
        )
        # for backwards compatibility
        return self.neighborhood_vehicle_states

    @staticmethod
    def _resolve_config(config, type_):
        if config is True:
            return type_()
        elif isinstance(config, type_):
            return config
        else:
            return False
