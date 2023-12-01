# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
from typing import Optional, Sequence

import numpy as np
from cached_property import cached_property
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from smarts.core.coordinates import Dimensions, Pose
from smarts.core.physics.collider import ColliderBase


class Chassis(ColliderBase):
    """Represents a vehicle chassis."""

    def control(self, *args, **kwargs):
        """Apply control values to the chassis."""
        raise NotImplementedError

    def reapply_last_control(self):
        """Re-apply the last given control given to the chassis."""
        raise NotImplementedError

    def teardown(self):
        """Clean up resources."""
        raise NotImplementedError

    @property
    def bullet_id(self) -> str:
        """The physics id of the chassis physics body."""
        raise NotImplementedError

    def set_pose(self, pose: Pose):
        """Use with caution since it disrupts the physics simulation. Sets the pose of the
        chassis.
        """
        raise NotImplementedError

    @property
    def steering(self) -> Optional[float]:
        """The steering value of the chassis in radians [-math.pi:math.pi]."""
        raise NotImplementedError

    def inherit_physical_values(self, other: "Chassis"):
        """Apply GCD between the two chassis."""
        raise NotImplementedError

    @property
    def to_polygon(self) -> Polygon:
        """Convert the chassis to a 2D shape."""
        p = self.pose.as_position2d()
        d = self.dimensions
        poly = shapely_box(
            p[0] - d.width * 0.5,
            p[1] - d.length * 0.5,
            p[0] + d.width * 0.5,
            p[1] + d.length * 0.5,
        )
        return shapely_rotate(poly, self.pose.heading, use_radians=True)

    def step(self, current_simulation_time):
        """Update the chassis state."""
        raise NotImplementedError


class BoxChassis(Chassis):
    """Control a vehicle by setting its absolute position and heading. The collision
    shape of the vehicle is a box of the provided dimensions.
    """

    def control(self, pose: Pose, speed: float, dt: float = 0):
        raise NotImplementedError

    def reapply_last_control(self):
        # no need to do anything here since we're not applying forces
        raise NotImplementedError

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """Use with care!  In essence, this is tinkering with the physics of the world,
        and may have unintended behavioral or performance consequences."""
        raise NotImplementedError

    def set_pose(self, pose: Pose):
        raise NotImplementedError

    @property
    def dimensions(self) -> Dimensions:
        raise NotImplementedError

    @property
    def contact_points(self) -> Sequence:
        raise NotImplementedError

    @property
    def bullet_id(self) -> str:
        raise NotImplementedError

    @property
    def speed(self) -> float:
        raise NotImplementedError

    @property
    def velocity_vectors(self):
        raise NotImplementedError

    @speed.setter
    def speed(self, speed: Optional[float] = None):
        raise NotImplementedError

    @property
    def pose(self) -> Pose:
        raise NotImplementedError

    @property
    def steering(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def yaw_rate(self) -> Optional[float]:
        raise NotImplementedError

    def inherit_physical_values(self, other: Chassis):
        raise NotImplementedError

    def step(self, current_simulation_time):
        raise NotImplementedError

    def teardown(self):
        raise NotImplementedError


class AckermannChassis(Chassis):
    """Control a vehicle by applying forces on its joints. The joints and links are
    defined by a URDF file.
    """

    @property
    def pose(self) -> Pose:
        raise NotImplementedError

    def set_pose(self, pose: Pose):
        raise NotImplementedError

    @property
    def steering(self):
        raise NotImplementedError

    @property
    def speed(self) -> float:
        raise NotImplementedError

    @property
    def velocity_vectors(self):
        raise NotImplementedError

    @speed.setter
    def speed(self, speed: Optional[float] = None):
        raise NotImplementedError

    @property
    def yaw_rate(self) -> float:
        """Returns 2-D rotational speed in rad/sec."""
        raise NotImplementedError

    @cached_property
    def longitudinal_lateral_speed(self):
        """Returns speed in m/s."""
        raise NotImplementedError

    @property
    def front_rear_stiffness(self):
        """The front and rear stiffness values of the tires on this chassis."""
        raise NotImplementedError

    @property
    def approx_max_speed(self):
        """This is the scientifically discovered maximum speed of this vehicle model"""
        raise NotImplementedError

    @property
    def contact_points(self):
        raise NotImplementedError

    @cached_property
    def mass_and_inertia(self):
        """The mass and inertia values of this chassis."""
        raise NotImplementedError

    @property
    def controller_parameters(self):
        """The current controller parameters for this chassis."""
        raise NotImplementedError

    @property
    def dimensions(self):
        raise NotImplementedError

    @property
    def max_steering_wheel(self):
        """Maximum steering output for the current gear ratio."""
        raise NotImplementedError

    @property
    def wheel_radius(self):
        """The wheel radius of the wheels on the chassis."""
        raise NotImplementedError

    @property
    def front_rear_axle_CG_distance(self):
        """The axle offsets from the vehicle base."""
        raise NotImplementedError

    @property
    def front_track_width(self):
        """The track width between the front wheels."""
        raise NotImplementedError

    @property
    def rear_track_width(self):
        """The track width between the back wheels."""
        raise NotImplementedError

    @property
    def max_torque(self):
        """The maximum throttle torque."""
        raise NotImplementedError

    @property
    def max_btorque(self):
        """The maximum break torque."""
        raise NotImplementedError

    @property
    def steering_ratio(self):
        """The steering gear ratio"""
        raise NotImplementedError

    @property
    def bullet_id(self) -> str:
        raise NotImplementedError

    @property
    def bullet_client(self):
        """The bullet physics simulator."""
        raise NotImplementedError

    def step(self, current_simulation_time):
        raise NotImplementedError

    def inherit_physical_values(self, other: BoxChassis):
        raise NotImplementedError

    def teardown(self):
        raise NotImplementedError

    def control(self, throttle=0, brake=0, steering=0):
        raise NotImplementedError

    def reapply_last_control(self):
        raise NotImplementedError

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        raise NotImplementedError
