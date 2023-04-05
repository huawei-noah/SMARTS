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
class DistCompleted:
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
class Time:
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
    collisions:Collisions= Collisions()
    comfort:Comfort = Comfort()
    dist_completed:DistCompleted = DistCompleted()
    dist_to_obstacles:DistToObstacles = DistToObstacles()
    gap_between_vehicles:GapBetweenVehicles = GapBetweenVehicles()
    jerk_linear:JerkLinear = JerkLinear()
    lane_center_offset:LaneCenterOffset = LaneCenterOffset()
    off_road:OffRoad = OffRoad()
    speed_limit:SpeedLimit = SpeedLimit()
    time:Time = Time()
    wrong_way:WrongWay = WrongWay()
