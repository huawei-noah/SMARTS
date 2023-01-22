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
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, SupportsFloat, Tuple, Union

import numpy as np
from cached_property import cached_property
from shapely.geometry import Point as SPoint
from typing_extensions import SupportsIndex

from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    radians_to_vec,
    yaw_from_quaternion,
)


class Dimensions(NamedTuple):
    """Representation of the size of a 3-dimensional form."""

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


_numpy_points = {}
_shapely_points = {}


class Point(NamedTuple):
    """A coordinate in space."""

    x: float
    y: float
    z: Optional[float] = 0

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        """Factory for constructing a Point object from a numpy array."""
        assert 2 <= len(np_array) <= 3
        z = np_array[2] if len(np_array) > 2 else 0.0
        return cls(np_array[0], np_array[1], z)

    @property
    def as_np_array(self) -> np.ndarray:
        """Convert this Point to a read-only numpy array and cache the result."""
        # Since this happens frequently and numpy array construction
        # involves memory allocation, we include this convenience method
        # with a cache of the result.
        # Note that before python3.8, @cached_property was not thread safe,
        # nor can it be used in a NamedTuple (which doesn't have a __dict__).
        # (Points can be used by multi-threaded client code, even when
        # SMARTS is still single-threaded, so we want to be safe here.)
        # So we use the private global _numpy_points as a cache instead.
        # Here we are relying on CPython's implementation of dict
        # to be thread-safe.
        cached = _numpy_points.get(self)
        if cached is not None:
            return cached
        npt = np.array((self.x, self.y, self.z))
        # the array shouln't be changed independently of this Point object now...
        npt.setflags(write=False)
        _numpy_points[self] = npt
        return npt

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
        if _numpy_points and self in _numpy_points:
            del _numpy_points[self]


class RefLinePoint(NamedTuple):
    """A reference line coordinate.
    See the Reference Line coordinate system in OpenDRIVE here:
    `https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09#_coordinate_systems`
    Also known as the Frenet coordinate system.
    """

    s: float  # offset along lane from start of lane
    t: Optional[float] = 0  # horizontal displacement from center of lane
    h: Optional[float] = 0  # vertical displacement from surface of lane


@dataclass(frozen=True)
class BoundingBox:
    """A 2-dimensional axis aligned box located in a [x, y] coordinate system."""

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

    def contains(self, pt: Point) -> bool:
        """returnx True iff pt is fully within the bounding box.  If any bbox coordinates are None, it's considered unbounded on that dimension/axis."""
        return (
            self.min_pt is None
            or (self.min_pt.x is None or self.min_pt.x < pt.x)
            and (self.min_pt.y is None or self.min_pt.y < pt.y)
        ) and (
            self.max_pt is None
            or (self.max_pt.x is None or pt.x < self.max_pt.x)
            and (self.max_pt.y is None or pt.y < self.max_pt.y)
        )


class Heading(float):
    """In this space we use radians, 0 is facing north, and turn counter-clockwise."""

    def __init__(self, value=...):
        float.__init__(value)

    def __new__(self, x: Union[SupportsFloat, SupportsIndex, Ellipsis.__class__] = ...):
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
        """Bullet's space is in radians, 0 faces north, and turns
        counter-clockwise.
        """
        h = Heading(bullet_heading)
        h.source = "bullet"
        return h

    @classmethod
    def from_panda3d(cls, p3d_heading):
        """Panda3D's space is in degrees, 0 faces north,
        and turns counter-clockwise.
        """
        h = Heading(math.radians(p3d_heading))
        h.source = "p3d"
        return h

    @classmethod
    def from_sumo(cls, sumo_heading):
        """Sumo's space uses degrees, 0 faces north, and turns clockwise."""
        heading = Heading.flip_clockwise(math.radians(sumo_heading))
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
        return math.degrees(Heading.flip_clockwise(self))

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
    def flip_clockwise(x):
        """Converts clockwise to counter-clockwise, and vice-versa."""
        return (2 * math.pi - x) % (2 * math.pi)

    def __repr__(self):
        return f"Heading({super().__repr__()})"


@dataclass
class Pose:
    """A pair of position and orientation values."""

    position: np.ndarray  # [x, y, z]
    """Center of vehicle."""
    orientation: np.ndarray  # [a, b, c, d] -> a + bi + cj + dk = 0
    heading_: Optional[Heading] = None  # cached heading to avoid recomputing

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)
        assert len(self.position) <= 3
        if len(self.position) < 3:
            self.position = np.resize(self.position, 3)
        assert len(self.orientation) == 4
        if not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation, dtype=np.float64)

    def __eq__(self, other: Any) -> bool:
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
    def from_front_bumper(cls, front_bumper_position, heading, length) -> "Pose":
        """Convert from front bumper location to a Pose with center of vehicle.

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
    def from_center(cls, base_position, heading: Heading):
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
        cls,
        offset_from_centre,
        base_position: np.ndarray,
        heading: Heading,
        local_heading: Heading,
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

    def as_position2d(self) -> np.ndarray:
        """Convert to a 2d position array"""
        return self.position[:2]

    def as_panda3d(self):
        """Convert to panda3D (object bounds centre position, heading)"""
        return (self.position, self.heading.as_panda3d)

    @classmethod
    def origin(cls):
        """Pose at the origin coordinate of smarts."""
        return cls(np.repeat([0], 3), np.array([0, 0, 0, 1]))
