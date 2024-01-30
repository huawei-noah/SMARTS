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
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from smarts.core.utils.cache import cache

if TYPE_CHECKING:
    from smarts.core import plan, signals
    from smarts.core.coordinates import Dimensions, Heading, Point, RefLinePoint
    from smarts.core.events import Events
    from smarts.core.road_map import Waypoint


class VehicleObservation(NamedTuple):
    """Perceived vehicle information."""

    id: str
    """The vehicle identifier."""
    position: Tuple[float, float, float]
    """The position of the vehicle within the simulation."""
    bounding_box: Dimensions
    """A bounding box describing the extents of the vehicle."""
    heading: Heading
    """The facing direction of the vehicle."""
    speed: float
    """The travel m/s in the direction of the vehicle."""
    road_id: str
    """The identifier for the road nearest to this vehicle."""
    lane_id: str
    """The identifier for the lane nearest to this vehicle."""
    lane_index: int
    """The index of the nearest lane on the road nearest to this vehicle."""
    lane_position: Optional[RefLinePoint] = None
    """(s,t,h) coordinates within the lane, where s is the longitudinal offset along the lane, t is the lateral displacement from the lane center, and h (not yet supported) is the vertical displacement from the lane surface.
    See the Reference Line coordinate system in OpenDRIVE here: https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09#_coordinate_systems """
    interest: bool = False
    """If this vehicle is of interest in the current scenario."""


class EgoVehicleObservation(NamedTuple):
    """Perceived ego vehicle information."""

    id: str
    """Vehicle identifier."""
    position: Tuple[float, float, float]
    """Center coordinate of the vehicle bounding box's bottom plane."""
    bounding_box: Dimensions
    """Bounding box describing the length, width, and height, of the vehicle."""
    heading: Heading
    """Facing direction of the vehicle. Units=rad."""
    speed: float
    """Travel speed in the direction of the vehicle. Units=m/s."""
    steering: float
    """Angle of front wheels in radians between [-pi, pi]."""
    yaw_rate: float
    """Speed of vehicle-heading rotation about the z-axis. Equivalent scalar representation of angular_velocity. Units=rad/s."""
    road_id: str
    """Identifier for the road nearest to this vehicle."""
    lane_id: str
    """Identifier for the lane nearest to this vehicle."""
    lane_index: int
    """Index of the nearest lane on the road nearest to this vehicle. Right most lane has index 0 and index increments to the left."""
    mission: plan.NavigationMission
    """Vehicle's desired destination."""
    linear_velocity: Tuple[float, float, float]
    """Velocity of vehicle along the global coordinate axes. Units=m/s."""
    angular_velocity: Tuple[float, float, float]
    """Velocity of vehicle-heading rotation about the z-axis. Equivalent vector representation of yaw_rate. Units=rad/s."""
    linear_acceleration: Optional[Tuple[float, float, float]]
    """Acceleration of vehicle along the global coordinate axes. Units=m/s^2. Requires accelerometer sensor."""
    angular_acceleration: Optional[Tuple[float, float, float]]
    """Acceleration of vehicle-heading rotation about the z-axis. Units=rad/s^2. Requires accelerometer sensor."""
    linear_jerk: Optional[Tuple[float, float, float]]
    """Jerk of vehicle along the global coordinate axes. Units=m/s^3. Requires accelerometer sensor."""
    angular_jerk: Optional[Tuple[float, float, float]]
    """Jerk of vehicle-heading rotation about the z-axis. Units=rad/s^3. Requires accelerometer sensor."""
    lane_position: Optional[RefLinePoint] = None
    """(s,t,h) coordinates within the lane, where s is the longitudinal offset along the lane, t is the lateral displacement from the lane center, and h (not yet supported) is the vertical displacement from the lane surface.
    See the Reference Line coordinate system in OpenDRIVE here: https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09#_coordinate_systems """


class RoadWaypoints(NamedTuple):
    """Per-road waypoint information."""

    lanes: Dict[str, List[List[Waypoint]]]
    """Mapping of road ids to their lane waypoints."""

    def __hash__(self) -> int:
        return hash(tuple((k, len(v)) for k, v in self.lanes.items()))


class GridMapMetadata(NamedTuple):
    """Map grid metadata."""

    resolution: float
    """Map resolution in world-space-distance/cell."""
    width: int
    """Map width in # of cells."""
    height: int
    """Map height in # of cells."""
    camera_position: Tuple[float, float, float]
    """Camera position when projected onto the map."""
    camera_heading: float
    """Camera rotation angle along z-axis when projected onto the map."""


class TopDownRGB(NamedTuple):
    """RGB camera observation."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """A RGB image with the ego vehicle at the center."""

    def __hash__(self) -> int:
        return self.metadata.__hash__()


class OccupancyGridMap(NamedTuple):
    """Occupancy map."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """An occupancy grid map around the ego vehicle. 
    
    See https://en.wikipedia.org/wiki/Occupancy_grid_mapping."""

    def __hash__(self) -> int:
        return self.metadata.__hash__()


class OcclusionRender(NamedTuple):
    """Occlusion map."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """A map showing what is visible from the ego vehicle"""

    def __hash__(self) -> int:
        return self.metadata.__hash__()


class DrivableAreaGridMap(NamedTuple):
    """Drivable area map."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """A grid map that shows the static drivable area around the ego vehicle."""

    def __hash__(self) -> int:
        return self.metadata.__hash__()


class CustomRenderData(NamedTuple):
    """Describes information about a custom render."""

    metadata: GridMapMetadata
    """Render metadata."""
    data: np.ndarray
    """The image data from the render."""

    def __hash__(self) -> int:
        return self.metadata.__hash__()


class ViaPoint(NamedTuple):
    """'Collectibles' that can be placed within the simulation."""

    position: Tuple[float, float]
    """Location (x,y) of this collectible."""
    lane_index: float
    """Lane index on the road this collectible is associated with."""
    road_id: str
    """Road id this collectible is associated with."""
    required_speed: float
    """Approximate speed required to collect this collectible."""
    hit: bool
    """If this via point was hit in the last step."""


class Vias(NamedTuple):
    """Listing of nearby collectible ViaPoints and ViaPoints collected in the last step."""

    near_via_points: Tuple[ViaPoint]
    """Ordered list of nearby points that have not been hit."""

    @property
    def hit_via_points(self) -> Tuple[ViaPoint]:
        """List of points that were hit in the previous step."""
        return tuple(vp for vp in self.near_via_points if vp.hit)


class SignalObservation(NamedTuple):
    """Describes an observation of a traffic signal (light) on this time-step."""

    state: signals.SignalLightState
    """The state of the traffic signal."""
    stop_point: Point
    """The stopping point for traffic controlled by the signal, i.e., the
    point where actors should stop when the signal is in a stop state."""
    controlled_lanes: Tuple[str]
    """If known, the lane_ids of all lanes controlled-by this signal.
    May be empty if this is not easy to determine."""
    last_changed: Optional[float]
    """If known, the simulation time this signal last changed its state."""


class Observation(NamedTuple):
    """The simulation observation."""

    dt: float
    """Amount of simulation time the last step took."""
    step_count: int
    """Number of steps taken by SMARTS thus far in the current scenario."""
    steps_completed: int
    """Number of steps this agent has taken within SMARTS."""
    elapsed_sim_time: float
    """Amount of simulation time elapsed for the current scenario."""
    events: Events
    """Classified observations that can trigger agent done status."""
    ego_vehicle_state: EgoVehicleObservation
    """Ego vehicle status."""
    under_this_agent_control: bool
    """Whether this agent currently has control of the vehicle."""
    neighborhood_vehicle_states: Optional[Tuple[VehicleObservation]]
    """List of neighborhood vehicle states."""
    waypoint_paths: Optional[List[List[Waypoint]]]
    """Dynamic evenly-spaced points on the road ahead of the vehicle, showing potential routes ahead."""
    distance_travelled: float
    """Road distance driven by the vehicle."""
    road_waypoints: Optional[RoadWaypoints]
    """Per-road waypoints information."""
    via_data: Vias
    """Listing of nearby collectible ViaPoints and ViaPoints collected in the last step."""
    # TODO: Convert to `NamedTuple` or only return point cloud.
    lidar_point_cloud: Optional[
        Tuple[List[np.ndarray], List[bool], List[Tuple[np.ndarray, np.ndarray]]]
    ] = None
    """Lidar point cloud consisting of [points, hits, (ray_origin, ray_vector)]. 
    Points missed (i.e., not hit) have `inf` value."""
    drivable_area_grid_map: Optional[DrivableAreaGridMap] = None
    """Drivable area map."""
    occupancy_grid_map: Optional[OccupancyGridMap] = None
    """Occupancy map."""
    top_down_rgb: Optional[TopDownRGB] = None
    """RGB camera observation."""
    signals: Optional[Tuple[SignalObservation]] = None
    """List of nearby traffic signal (light) states on this time-step."""
    occlusion_map: Optional[OcclusionRender] = None
    """Observable area map."""
    custom_renders: Tuple[CustomRenderData, ...] = tuple()
    """Custom renders."""

    def __hash__(self):
        return hash(
            (
                self.dt,
                self.step_count,
                self.elapsed_sim_time,
                self.events,
                self.ego_vehicle_state,
                # self.waypoint_paths, # likely redundant
                self.neighborhood_vehicle_states,
                self.distance_travelled,
                self.road_waypoints,
                self.via_data,
                self.drivable_area_grid_map,
                self.occupancy_grid_map,
                self.top_down_rgb,
                self.signals,
                self.occlusion_map,
                self.custom_renders,
            )
        )
