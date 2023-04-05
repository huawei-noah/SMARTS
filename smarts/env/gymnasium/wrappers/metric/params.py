from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Collisions:
    active: bool = False


@dataclass
class Comfort:
    active: bool = False


@dataclass
class DistCompleted:
    active: bool = True
    wrt: str = "self"


@dataclass
class DistToObstacles:
    active: bool = True
    ignore: Optional[List[str]] = None


@dataclass
class GapBetweenVehicles:
    active: bool = False
    interest: str = "Leader-007"


@dataclass
class JerkLinear:
    active: bool = True


@dataclass
class LaneCenterOffset:
    active: bool = True


@dataclass
class OffRoad:
    active: bool = False


@dataclass
class SpeedLimit:
    active: bool = True


@dataclass
class Time:
    active: bool = True


@dataclass
class WrongWay:
    active: bool = True


@dataclass
class Params:
    collisions = Collisions()
    comfort = Comfort()
    dist_completed = DistCompleted()
    dist_to_obstacles = DistToObstacles()
    gap_between_vehicles = GapBetweenVehicles()
    jerk_linear = JerkLinear()
    lane_center_offset = LaneCenterOffset()
    off_road = OffRoad()
    speed_limit = SpeedLimit()
    time = Time()
    wrong_way = WrongWay()
