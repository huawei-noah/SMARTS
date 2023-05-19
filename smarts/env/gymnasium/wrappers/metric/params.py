# MIT License

# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Collisions:
    """Parameters for collisions cost function."""

    active: bool = False
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class Comfort:
    """Parameters for comfort cost function."""

    active: bool = False
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class DistToDestination:
    """Parameters for distance to destination cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class DistToObstacles:
    """Parameters for distance to obstacles cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """
    ignore: List[str] = field(default_factory=list)
    """Vehicle id of neighbors to be excluded from being considered as an
    obstacle.
    """


@dataclass(frozen=True)
class JerkLinear:
    """Parameters for jerk linear cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class LaneCenterOffset:
    """Parameters for lane center offset cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class OffRoad:
    """Parameters for off road cost function."""

    active: bool = False
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class SpeedLimit:
    """Parameters for speed limit cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class Steps:
    """Parameters for steps cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class VehicleGap:
    """Parameters for gap between vehicles cost function."""

    active: bool = False
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """
    actor: str = ""
    """Gap between ego and the specified `actor` is computed."""


@dataclass(frozen=True)
class WrongWay:
    """Parameters for wrong way cost function."""

    active: bool = True
    """If True, enables computation of corresponding cost function. Else, 
    disabled.
    """


@dataclass(frozen=True)
class Params:
    """Parameters for cost functions."""

    collisions: Collisions = Collisions()
    comfort: Comfort = Comfort()
    dist_to_destination: DistToDestination = DistToDestination()
    dist_to_obstacles: DistToObstacles = DistToObstacles()
    jerk_linear: JerkLinear = JerkLinear()
    lane_center_offset: LaneCenterOffset = LaneCenterOffset()
    off_road: OffRoad = OffRoad()
    speed_limit: SpeedLimit = SpeedLimit()
    steps: Steps = Steps()
    vehicle_gap: VehicleGap = VehicleGap()
    wrong_way: WrongWay = WrongWay()
