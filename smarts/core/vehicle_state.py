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
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from smarts.core.coordinates import Dimensions

from .actor import ActorState
from .colors import SceneColors
from .coordinates import Dimensions, Heading, Pose
from .utils.core_math import rotate_cw_around_point


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle configuration"""

    vehicle_type: str
    color: SceneColors
    dimensions: Dimensions
    glb_model: str


# A mapping between SUMO's vehicle types and our internal vehicle config.
# TODO: Don't leak SUMO's types here.
# XXX: The GLB's dimensions must match the specified dimensions here.
# TODO: for traffic histories, vehicle (and road) dimensions are set by the dataset
VEHICLE_CONFIGS = {
    "passenger": VehicleConfig(
        vehicle_type="car",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=3.68, width=1.47, height=1.4),
        glb_model="simple_car.glb",
    ),
    "bus": VehicleConfig(
        vehicle_type="bus",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=7, width=2.25, height=3),
        glb_model="bus.glb",
    ),
    "coach": VehicleConfig(
        vehicle_type="coach",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=8, width=2.4, height=3.5),
        glb_model="coach.glb",
    ),
    "truck": VehicleConfig(
        vehicle_type="truck",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=5, width=1.91, height=1.89),
        glb_model="truck.glb",
    ),
    "trailer": VehicleConfig(
        vehicle_type="trailer",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=10, width=2.5, height=4),
        glb_model="trailer.glb",
    ),
    "pedestrian": VehicleConfig(
        vehicle_type="pedestrian",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=0.5, width=0.5, height=1.6),
        glb_model="pedestrian.glb",
    ),
    "motorcycle": VehicleConfig(
        vehicle_type="motorcycle",
        color=SceneColors.SocialVehicle,
        dimensions=Dimensions(length=2.5, width=1, height=1.4),
        glb_model="motorcycle.glb",
    ),
}


class Collision(NamedTuple):
    """Represents a collision by an ego vehicle with another vehicle."""

    # XXX: This might not work for boid agents
    collidee_id: str
    """The id of the body that was collided with."""
    collidee_owner_id: str
    """The id of the controlling agent or other controlling entity."""


@dataclass
class VehicleState(ActorState):
    """Vehicle state information."""

    vehicle_config_type: Optional[str] = None  # key into VEHICLE_CONFIGS
    pose: Optional[Pose] = None
    dimensions: Optional[Dimensions] = None
    speed: float = 0.0
    steering: Optional[float] = None
    yaw_rate: Optional[float] = None
    linear_velocity: Optional[np.ndarray] = None
    angular_velocity: Optional[np.ndarray] = None
    linear_acceleration: Optional[np.ndarray] = None
    angular_acceleration: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.pose is not None and self.dimensions is not None

    def __eq__(self, __o: object):
        return (
            isinstance(__o, type(self))
            and super().__eq__(__o)
            and self.pose == __o.pose
        )

    def get_pose(self) -> Optional[Pose]:
        return self.pose

    def get_dimensions(self) -> Optional[Dimensions]:
        return self.dimensions

    def linear_velocity_tuple(self) -> Optional[Tuple[float, float, float]]:
        """Generates a tuple representation of linear velocity with standard python types."""
        return (
            None
            if self.linear_velocity is None
            else tuple(float(f) for f in self.linear_velocity)
        )

    def angular_velocity_tuple(self) -> Optional[Tuple[float, float, float]]:
        """Generates a tuple representation of angular velocity with standard python types."""
        return (
            None
            if self.angular_velocity is None
            else tuple(float(f) for f in self.angular_velocity)
        )

    @property
    def bounding_box_points(
        self,
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        """The minimum fitting heading aligned bounding box. Four 2D points representing the minimum fitting box."""
        # Assuming the position is the center,
        # calculate the corner coordinates of the bounding_box
        origin = self.pose.position[:2]
        dimensions = np.array([self.dimensions.width, self.dimensions.length])
        corners = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
        heading = self.pose.heading
        return tuple(
            tuple(
                rotate_cw_around_point(
                    point=origin + corner * dimensions,
                    radians=Heading.flip_clockwise(heading),
                    origin=origin,
                )
            )
            for corner in corners
        )

    @property
    def bbox(self) -> Polygon:
        """Returns a bounding box around the vehicle."""
        pos = self.pose.point
        half_len = 0.5 * self.dimensions.length
        half_width = 0.5 * self.dimensions.width
        poly = shapely_box(
            pos.x - half_width,
            pos.y - half_len,
            pos.x + half_width,
            pos.y + half_len,
        )
        return shapely_rotate(poly, self.pose.heading, use_radians=True)


def neighborhood_vehicles_around_vehicle(
    vehicle_state, vehicle_states, radius: Optional[float] = None
):
    """Determines what vehicles are within the radius (if given)."""
    other_states = [v for v in vehicle_states if v.actor_id != vehicle_state.actor_id]
    if radius is None:
        return other_states

    other_positions = [state.pose.position for state in other_states]
    if not other_positions:
        return []

    # calculate euclidean distances
    distances = np.linalg.norm(other_positions - vehicle_state.pose.position, axis=1)

    indices = np.argwhere(distances <= radius).flatten()
    return [other_states[i] for i in indices]
