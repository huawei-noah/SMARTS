# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import enum
import math
from dataclasses import dataclass
from typing import Optional, Sequence, SupportsFloat, Union, NamedTuple

import numpy as np
from typing_extensions import SupportsIndex

from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    radians_to_vec,
    yaw_from_quaternion,
)


class BoundingBox(NamedTuple):
    length: float = None
    width: float = None
    height: float = None


class Heading(float):
    """In this space we use radians, 0 is facing north, and turn counter-clockwise."""

    def __init__(self, value=...):
        pass

    def __new__(self, x: Union[SupportsFloat, SupportsIndex] = ...) -> "Heading":
        """A override to constrain heading to -pi to pi"""
        value = x
        if isinstance(value, (int, float)):
            value = value % (2 * math.pi)
            if value > math.pi:
                value -= 2 * math.pi
        if x in {..., None}:
            value = 0
        return float.__new__(self, value)

    def __repr__(self):
        return f"Heading({super().__repr__()})"


class HeadingMethods:
    @classmethod
    def from_bullet(cls, bullet_heading):
        """Bullet's space is in radians, 0 faces north, and we turn
        counter-clockwise.
        """
        h = Heading(bullet_heading)
        return h

    @classmethod
    def from_panda3d(cls, p3d_heading):
        """Panda3D's space is in degrees, 0 faces north,
        and we turn counter-clockwise.
        """
        h = Heading(math.radians(p3d_heading))
        return h

    @classmethod
    def from_sumo(cls, sumo_heading):
        """Sumo's space uses degrees, 0 faces north, and we turn clockwise."""
        heading = HeadingMethods._flip_clockwise(math.radians(sumo_heading))
        h = Heading(heading)
        return h

    @staticmethod
    def as_panda3d(heading):
        return math.degrees(heading)

    @staticmethod
    def as_bullet(heading):
        return heading

    @staticmethod
    def as_sumo(heading):
        return math.degrees(HeadingMethods._flip_clockwise(heading))

    @staticmethod
    def relative_to(heading, other: "Heading"):
        """
        Computes the relative heading w.r.t. the other given heading
        >>> HeadingMethods.relative_to(Heading(math.pi/4), Heading(math.pi))
        Heading(-2.356194490192345)
        """
        assert isinstance(other, Heading)

        rel_heading = Heading(heading - other)

        assert -math.pi <= rel_heading <= math.pi, f"{rel_heading}"

        return Heading(rel_heading)

    # 2D directional vector that aligns with Cartesian Coordinate System
    @staticmethod
    def direction_vector(heading):
        return radians_to_vec(heading)

    @staticmethod
    def _flip_clockwise(x):
        """Converts clockwise to counter-clockwise, and vice-versa."""
        return (2 * math.pi - x) % (2 * math.pi)


@dataclass
class Pose:
    # TODO: these should be np.ndarray
    position: Sequence  # [x, y, z]
    orientation: Sequence  # [a, b, c, d] -> a + bi + cj + dk = 0
    heading_: Optional[Heading] = None  # cached heading to avoid recomputing

    @classmethod
    def from_front_bumper(cls, front_bumper_position, heading, length):
        """Convert from front bumper location

        Args:
            front_bumper_position: The (x, y) position of the centre front of the front bumper
            heading: The heading of the pose
            length: The length dimension of the object's physical bounds
        """
        assert isinstance(front_bumper_position, np.ndarray)
        assert front_bumper_position.shape == (2,), f"{front_bumper_position.shape}"

        _orientation = fast_quaternion_from_angle(heading)
        lwh_offset = radians_to_vec(heading) * (0.5 * length)
        pos_2d = front_bumper_position - lwh_offset
        return cls(
            position=np.array([pos_2d[0], pos_2d[1], 0]),
            orientation=_orientation,
            heading_=heading,
        )

    @classmethod
    def from_center(cls, base_position, heading):
        """Convert from centred location

        Args:
            base_position: The center of the object's bounds
            heading: The heading of the object
        """
        assert isinstance(heading, Heading)

        position = np.array([*base_position, 0][:3])
        orientation = fast_quaternion_from_angle(heading)

        return cls(
            position=position,
            orientation=orientation,
            heading_=heading,
        )

    @classmethod
    def from_explicit_offset(
        cls, offset_from_centre, base_position, heading, local_heading
    ):
        """Convert from an explicit offset

        Args:
            offset_from_centre: The offset away from the centre of the object's bounds
            heading: The heading of the pose
            base_position: The base position without offset
            local_heading: An additional orientation that re-faces the center offset
        """
        assert isinstance(heading, Heading)
        assert isinstance(base_position, np.ndarray)

        orientation = fast_quaternion_from_angle(heading)
        oprime = heading + local_heading
        # Calculate rotation on xy-plane only, given that fast_quaternion_from_angle is also on xy-plane
        vprime = np.array(
            [
                offset_from_centre[0] * np.cos(oprime)
                - offset_from_centre[1] * np.sin(oprime),
                offset_from_centre[0] * np.sin(oprime)
                + offset_from_centre[1] * np.cos(oprime),
                offset_from_centre[2],
            ]
        )
        position = base_position + vprime
        return cls(position=position, orientation=orientation, heading_=heading)

    def as_sumo(self, length, local_heading):
        """Convert to SUMO (position of front bumper, cw_heading)

        args:
            heading:
                The heading of the pose
            length:
                The length dimension of the object's physical bounds
            local_heading:
                An additional orientation that re-faces the length offset
        """
        vprime = radians_to_vec(self.heading + local_heading) * 0.5 * length

        return (
            np.array([self.position[0] + vprime[0], self.position[1] + vprime[1], 0]),
            HeadingMethods.as_sumo(self.heading),
        )

    def as_bullet(self):
        """Convert to bullet origin (position of bullet origin, orientation quaternion"""
        return (self.position, self.orientation)

    @property
    def heading(self):
        if self.heading_ is None:
            yaw = yaw_from_quaternion(self.orientation)
            self.heading_ = Heading(yaw)

        return self.heading_

    def as_panda3d(self):
        """ Convert to panda3D (object bounds centre position, heading)"""
        return (self.position, HeadingMethods.as_panda3d(self.heading))
