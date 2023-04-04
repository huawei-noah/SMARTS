from dataclasses import dataclass, field

@dataclass
class DistCompleted:
    w: float = 0.35
    wrt: str = "self"

@dataclass
class Time:
    w: float = 0.25

@dataclass
class DistToObstacles:
    w: float = 0.15
    ignore: list = field(default_factory=list)

@dataclass
class Gap:
    w: float = 0

@dataclass
class JerkLinear:
    w: float = 0.15

@dataclass
class LaneCenterOffset:
    w: float = 0.1

@dataclass
class SpeedLimit:
    w: float = 0

@dataclass
class WrongWay:
    w: float = 0

@dataclass
class MetricParams:
    dist_completed: DistCompleted = DistCompleted()
    time: Time = Time()
    dist_to_obstacles: DistToObstacles = DistToObstacles()
    gap: Gap = Gap()
    jerk_linear : JerkLinear = JerkLinear()
    lane_center_offset: LaneCenterOffset = LaneCenterOffset()
    speed_limit : SpeedLimit = SpeedLimit()
    wrong_way: WrongWay = WrongWay()

def dict_to_class(recv:dict) -> MetricParams:
    params = MetricParams()
    for key1, value1 in recv.items():
        for key2, value2 in value1.items():
            obj = getattr(params,key1)
            assert getattr(obj,key2)
            setattr(obj, key2, value2)

    return params
