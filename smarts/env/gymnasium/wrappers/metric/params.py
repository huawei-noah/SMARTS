from dataclasses import dataclass, field, fields
from typing import Dict


 

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
            assert key2 in obj.__dataclass_fields__
            setattr(obj, key2, value2)

    return params

# def make_metrics_params(config:Dict):
#     from smarts.env.gymnasium.wrappers.metric.costs import CostFuncs

#     params = {}
#     for field in fields(CostFuncs):
#         if config.
#         params[field.name]
#     for key1, value1 in recv.items():
#         for key2, value2 in value1.items():
