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
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from smarts.core.coordinates import Dimensions, Heading, Point, RefLinePoint
from smarts.core.plan import Mission
from smarts.core.road_map import Waypoint
from smarts.core.signals import SignalLightState

from .events import Events


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


class EgoVehicleObservation(NamedTuple):
    """Perceived ego vehicle information."""

    id: str
    """Vehicle identifier."""
    position: np.ndarray
    """Center coordinate of the vehicle bounding box's bottom plane. shape=(3,). dtype=np.float64."""
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
    mission: Mission
    """Vehicle's desired destination."""
    linear_velocity: np.ndarray
    """Velocity of vehicle along the global coordinate axes. Units=m/s. A numpy array of shape=(3,) and dtype=np.float64."""
    angular_velocity: np.ndarray
    """Velocity of vehicle-heading rotation about the z-axis. Equivalent vector representation of yaw_rate. Units=rad/s. A numpy array of shape=(3,) and dtype=np.float64."""
    linear_acceleration: Optional[np.ndarray]
    """Acceleration of vehicle along the global coordinate axes. Units=m/s^2. A numpy array of shape=(3,). dtype=np.float64. Requires accelerometer sensor."""
    angular_acceleration: Optional[np.ndarray]
    """Acceleration of vehicle-heading rotation about the z-axis. Units=rad/s^2. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor."""
    linear_jerk: Optional[np.ndarray]
    """Jerk of vehicle along the global coordinate axes. Units=m/s^3. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor."""
    angular_jerk: Optional[np.ndarray]
    """Jerk of vehicle-heading rotation about the z-axis. Units=rad/s^3. A numpy array of shape=(3,) and dtype=np.float64. Requires accelerometer sensor."""
    lane_position: Optional[RefLinePoint] = None
    """(s,t,h) coordinates within the lane, where s is the longitudinal offset along the lane, t is the lateral displacement from the lane center, and h (not yet supported) is the vertical displacement from the lane surface.
    See the Reference Line coordinate system in OpenDRIVE here: https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09#_coordinate_systems """


class RoadWaypoints(NamedTuple):
    """Per-road waypoint information."""

    lanes: Dict[str, List[List[Waypoint]]]
    """Mapping of road ids to their lane waypoints."""


class GridMapMetadata(NamedTuple):
    """Map grid metadata."""

    created_at: int
    """Time at which the map was loaded."""
    resolution: float
    """Map resolution in world-space-distance/cell."""
    width: int
    """Map width in # of cells."""
    height: int
    """Map height in # of cells."""
    camera_position: Tuple[float, float, float]
    """Camera position when projected onto the map."""
    camera_heading_in_degrees: float
    """Camera rotation angle along z-axis when projected onto the map."""


class TopDownRGB(NamedTuple):
    """RGB camera observation."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """A RGB image with the ego vehicle at the center."""


class OccupancyGridMap(NamedTuple):
    """Occupancy map."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """An occupancy grid map around the ego vehicle. 
    
    See https://en.wikipedia.org/wiki/Occupancy_grid_mapping."""


class DrivableAreaGridMap(NamedTuple):
    """Drivable area map."""

    metadata: GridMapMetadata
    """Map metadata."""
    data: np.ndarray
    """A grid map that shows the static drivable area around the ego vehicle."""


class ViaPoint(NamedTuple):
    """'Collectables' that can be placed within the simulation."""

    position: Tuple[float, float]
    """Location (x,y) of this collectable."""
    lane_index: float
    """Lane index on the road this collectable is associated with."""
    road_id: str
    """Road id this collectable is associated with."""
    required_speed: float
    """Approximate speed required to collect this collectable."""


class Vias(NamedTuple):
    """Listing of nearby collectable ViaPoints and ViaPoints collected in the last step."""

    near_via_points: List[ViaPoint]
    """Ordered list of nearby points that have not been hit."""
    hit_via_points: List[ViaPoint]
    """List of points that were hit in the previous step."""


class SignalObservation(NamedTuple):
    """Describes an observation of a traffic signal (light) on this timestep."""

    state: SignalLightState
    """The state of the traffic signal."""
    stop_point: Point
    """The stopping point for traffic controlled by the signal, i.e., the
    point where actors should stop when the signal is in a stop state."""
    controlled_lanes: List[str]
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
    """Amout of simulation time elapsed for the current scenario."""
    events: Events
    """Classified observations that can trigger agent done status."""
    ego_vehicle_state: EgoVehicleObservation
    """Ego vehicle status."""
    under_this_agent_control: bool
    """Whether this agent currently has control of the vehicle."""
    neighborhood_vehicle_states: Optional[List[VehicleObservation]]
    """List of neighbourhood vehicle states."""
    waypoint_paths: Optional[List[List[Waypoint]]]
    """Dynamic evenly-spaced points on the road ahead of the vehicle, showing potential routes ahead."""
    distance_travelled: float
    """Road distance driven by the vehicle."""
    # TODO: Convert to `NamedTuple` or only return point cloud.
    lidar_point_cloud: Optional[
        Tuple[List[np.ndarray], List[bool], List[Tuple[np.ndarray, np.ndarray]]]
    ]
    """Lidar point cloud consisting of [points, hits, (ray_origin, ray_vector)]. 
    Points missed (i.e., not hit) have `inf` value."""
    drivable_area_grid_map: Optional[DrivableAreaGridMap]
    """Drivable area map."""
    occupancy_grid_map: Optional[OccupancyGridMap]
    """Occupancy map."""
    top_down_rgb: Optional[TopDownRGB]
    """RGB camera observation."""
    road_waypoints: Optional[RoadWaypoints]
    """Per-road waypoints information."""
    via_data: Vias
    """Listing of nearby collectable ViaPoints and ViaPoints collected in the last step."""
    signals: Optional[List[SignalObservation]] = None
    """List of nearby traffic signal (light) states on this timestep."""


class Collision(NamedTuple):
    """Represents a collision by an ego vehicle with another vehicle."""

    # XXX: This might not work for boid agents
    collidee_id: str
