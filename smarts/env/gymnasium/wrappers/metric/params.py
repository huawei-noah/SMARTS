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

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Collisions:
    active: bool = False
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class Comfort:
    active: bool = False
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class DistToDestination:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """
    wrt: str = "self"
    """ `wrt` specifies the name of the vehicle with respect to which the total
    distance is computed for an ego agent in a scenario. The total distance is 
    generally used as the denominator in distance completion metric. Options 
    include: 
    
    + "self" : Only available in scenarios where all egos have PositionalGoal 
    mission. Total distance represents the shortest road distance from the 
    ego's start position to its goal position.
    + SUMO/SMARTS vehicle id: Total distance represents the total road length 
    of the specified vehicle's route.
    """


@dataclass
class DistToObstacles:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """
    ignore: Optional[List[str]] = None


@dataclass
class GapBetweenVehicles:
    active: bool = False
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """
    interest: str = "Leader-007"


@dataclass
class JerkLinear:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class LaneCenterOffset:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class OffRoad:
    active: bool = False
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class SpeedLimit:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class Steps:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class WrongWay:
    active: bool = True
    """If True, enables computation of coresponding cost function. Else, 
    disabled.
    """


@dataclass
class Params:
    collisions: Collisions = Collisions()
    comfort: Comfort = Comfort()
    dist_to_destination: DistToDestination = DistToDestination()
    dist_to_obstacles: DistToObstacles = DistToObstacles()
    gap_between_vehicles: GapBetweenVehicles = GapBetweenVehicles()
    jerk_linear: JerkLinear = JerkLinear()
    lane_center_offset: LaneCenterOffset = LaneCenterOffset()
    off_road: OffRoad = OffRoad()
    speed_limit: SpeedLimit = SpeedLimit()
    steps: Steps = Steps()
    wrong_way: WrongWay = WrongWay()
