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
from enum import Enum
from typing import NamedTuple, Dict, Sequence, Tuple

from smarts.core.events import Events


class TrafficActorType(str, Enum):
    SocialVehicle = "social_vehicle"
    SocialAgent = "social_agent"
    Agent = "agent"


class VehicleType(str, Enum):
    Bus = "bus"
    Coach = "coach"
    Truck = "truck"
    Trailer = "trailer"
    Car = "car"


class TrafficActorState(NamedTuple):
    actor_type: TrafficActorType
    vehicle_type: VehicleType
    position: Tuple[float, float, float]
    heading: float
    speed: float
    name: str = ""
    actor_id: str = None
    events: Events = None
    waypoint_paths: Sequence = []
    driven_path: Sequence = []
    point_cloud: Sequence = []
    mission_route_geometry: Sequence[Sequence[Tuple[float, float]]] = None


class State(NamedTuple):
    traffic: Dict[str, TrafficActorState]
    scenario_id: str
    # sequence of x, y coordinates
    bubbles: Sequence[Sequence[Tuple[float, float]]]
    scene_colors: Dict[str, Tuple[float, float, float, float]]
    scores: Dict[str, float]


def format_actor_id(actor_id: str, vehicle_id: str, is_multi: bool):
    if is_multi:
        return f"{actor_id}-{{{vehicle_id[:4]}}}"
    return actor_id
