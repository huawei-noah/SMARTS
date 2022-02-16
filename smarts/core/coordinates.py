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
import enum
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, SupportsFloat, Tuple, Type, Union

import numpy as np
from cached_property import cached_property
from shapely.geometry import Point as SPoint
from typing_extensions import SupportsIndex

from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    radians_to_vec,
    yaw_from_quaternion,
)


@dataclass(frozen=True)
class Dimensions:
    """A 3 dimension data structure representing a box."""

    length: float
    width: float
    height: float

    @classmethod
    def init_with_defaults(
        cls, length: float, width: float, height: float, defaults: "Dimensions"
    ) -> "Dimensions":
        """Create with the given default values"""
        if not length or length == -1:
            length = defaults.length
        if not width or width == -1:
            width = defaults.width
        if not height or height == -1:
            height = defaults.height
        return cls(length, width, height)

    @classmethod
    def copy_with_defaults(
        cls, dims: "Dimensions", defaults: "Dimensions"
    ) -> "Dimensions":
        """Make a copy of the given dimensions with a default option."""
        return cls.init_with_defaults(dims.length, dims.width, dims.height, defaults)

    @property
    def as_lwh(self) -> Tuple[float, float, float]:
        """Convert to a tuple consisting of (length, width, height)."""
        return (self.length, self.width, self.height)

    def equal_if_defined(self, length: float, width: float, height: float) -> bool:
        """Test if dimensions are matching."""
        return (
            (not self.length or self.length == -1 or self.length == length)
            and (not self.width or self.width == -1 or self.width == width)
            and (not self.height or self.height == -1 or self.height == height)
        )


_shapely_points = {}


class Point(NamedTuple):
    """A coordinate in space."""

    x: float
    y: float
    z: Optional[float] = 0

    @property
    def as_shapely(self) -> SPoint:
        """Use with caution! Convert this point to a shapely point."""
        # Shapely Point construction is expensive!
        # Note that before python3.8, @cached_property was not thread safe,
        # nor can it be used in a NamedTuple (which doesn't have a __dict__).
        # (Points can be used by multi-threaded client code, even when
        # SMARTS is still single-threaded, so we want to be safe here.)
        # So we use the private global _shapely_points as a cache instead.
        # Here we are relying on CPython's implementation of dict
        # to be thread-safe.
        cached = _shapely_points.get(self)
        if cached:
            return cached
        spt = SPoint((self.x, self.y, self.z))
        _shapely_points[self] = spt
        return spt

    def __del__(self):
        if _shapely_points and self in _shapely_points:
            del _shapely_points[self]


class RefLinePoint(NamedTuple):
    """A reference line coordinate.
    See the Reference Line coordinate system in OpenDRIVE here:
       https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09#_coordinate_systems
    """

    s: float  # offset along lane from start of lane
    t: Optional[float] = 0  # horizontal displacement from center of lane
    h: Optional[float] = 0  # vertical displacement from surface of lane


@dataclass(frozen=True)
class BoundingBox:
    """A fitted box generally used to encapsulate geometry."""

    min_pt: Point
    max_pt: Point

    @property
    def length(self):
        """The length of the box."""
        return self.max_pt.x - self.min_pt.x

    @property
    def width(self):
        """The width of the box."""
        return self.max_pt.y - self.min_pt.y

    @property
    def height(self):
        """The height of the box."""
        return self.max_pt.z - self.min_pt.z

    @property
    def center(self):
        """The center point of the box."""
        return Point(
            x=(self.min_pt.x + self.max_pt.x) / 2,
            y=(self.min_pt.y + self.max_pt.y) / 2,
            z=(self.min_pt.z + self.max_pt.z) / 2,
        )

    @property
    def as_dimensions(self) -> Dimensions:
        """The box dimensions. This will lose offset information."""
        return Dimensions(length=self.length, width=self.width, height=self.height)


class Heading(float):
    """In this space we use radians, 0 is facing north, and turn counter-clockwise."""

    def __init__(self, value=...):
        float.__init__(value)

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

    @classmethod
    def from_bullet(cls, bullet_heading):
        """Bullet's space is in radians, 0 faces north, and we turn
        counter-clockwise.
        """
        h = Heading(bullet_heading)
        h.source = "bullet"
        return h

    @classmethod
    def from_panda3d(cls, p3d_heading):
        """Panda3D's space is in degrees, 0 faces north,
        and we turn counter-clockwise.
        """
        h = Heading(math.radians(p3d_heading))
        h.source = "p3d"
        return h

    @classmethod
    def from_sumo(cls, sumo_heading):
        """Sumo's space uses degrees, 0 faces north, and we turn clockwise."""
        heading = Heading._flip_clockwise(math.radians(sumo_heading))
        h = Heading(heading)
        h.source = "sumo"
        return h

    @property
    def as_panda3d(self):
        """Convert to Panda3D facing format."""
        return math.degrees(self)

    @property
    def as_bullet(self):
        """Convert to bullet physics facing format."""
        return self

    @property
    def as_sumo(self):
        """Convert to SUMO facing format"""
        return math.degrees(Heading._flip_clockwise(self))

    def relative_to(self, other: "Heading"):
        """
        Computes the relative heading w.r.t. the given heading
        >>> Heading(math.pi/4).relative_to(Heading(math.pi))
        Heading(-2.356194490192345)
        """
        assert isinstance(other, Heading)

        rel_heading = Heading(self - other)

        assert -math.pi <= rel_heading <= math.pi, f"{rel_heading}"

        return Heading(rel_heading)

    def direction_vector(self):
        """Convert to a 2D directional vector that aligns with Cartesian Coordinate System"""
        return radians_to_vec(self)

    @staticmethod
    def _flip_clockwise(x):
        """Converts clockwise to counter-clockwise, and vice-versa."""
        return (2 * math.pi - x) % (2 * math.pi)

    def __repr__(self):
        return f"Heading({super().__repr__()})"


@dataclass
class Pose:
    """A pair of position and orientation values."""

    # TODO: these should be np.ndarray
    position: Sequence  # [x, y, z]
    orientation: Sequence  # [a, b, c, d] -> a + bi + cj + dk = 0
    heading_: Optional[Heading] = None  # cached heading to avoid recomputing

    def __eq__(self, other) -> bool:
        if not isinstance(other, Pose):
            return False
        return (self.position == other.position).all() and (
            self.orientation == other.orientation
        ).all()

    def __hash__(self):
        return hash((*self.position, *self.orientation))

    def reset_with(self, position, heading: Heading):
        """Resets the pose with the given position and heading values."""
        if self.position.dtype is not np.dtype(np.float64):
            # The slice assignment below doesn't change self.position's dtype,
            # which can be a problem if it was initialized with ints and
            # now we are assigning it floats, so we just cast it...
            self.position = np.float64(self.position)
        self.position[:] = position
        if "point" in self.__dict__:
            # clear the cached_property
            del self.__dict__["point"]
        if heading != self.heading_:
            self.orientation = fast_quaternion_from_angle(heading)
            self.heading_ = heading

    @cached_property
    def point(self) -> Point:
        """The positional value of this pose as a point."""
        return Point(*self.position)

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
            self.heading.as_sumo,
        )

    def as_bullet(self):
        """Convert to bullet origin (position of bullet origin, orientation quaternion"""
        return (self.position, self.orientation)

    @property
    def heading(self):
        """The heading value converted from orientation."""

        # XXX: changing the orientation should invalidate this
        if self.heading_ is None:
            yaw = yaw_from_quaternion(self.orientation)
            self.heading_ = Heading(yaw)

        return self.heading_

    def as_panda3d(self):
        """ Convert to panda3D (object bounds centre position, heading)"""
        return (self.position, self.heading.as_panda3d)
