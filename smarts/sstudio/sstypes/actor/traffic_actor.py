# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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


from dataclasses import dataclass, field
from typing import Union

from smarts.core.utils.file import pickle_hash_int
from smarts.sstudio.sstypes.actor import Actor
from smarts.sstudio.sstypes.distribution import Distribution
from smarts.sstudio.sstypes.traffic_model import JunctionModel, LaneChangingModel


@dataclass(frozen=True)
class TrafficActor(Actor):
    """Used as a description/spec for traffic actors (e.x. Vehicles, Pedestrians,
    etc). The defaults provided are for a car, but the name is not set to make it
    explicit that you actually want a car.
    """

    accel: float = 2.6
    """The maximum acceleration value of the actor (in m/s^2)."""
    decel: float = 4.5
    """The maximum deceleration value of the actor (in m/s^2)."""
    tau: float = 1.0
    """The minimum time headway"""
    sigma: float = 0.5
    """The driver imperfection"""  # TODO: appears to not be used in generators.py
    depart_speed: Union[float, str] = "max"
    """The starting speed of the actor"""
    emergency_decel: float = 4.5
    """maximum deceleration ability of vehicle in case of emergency"""
    speed: Distribution = Distribution(mean=1.0, sigma=0.1)
    """The speed distribution of this actor in m/s."""
    imperfection: Distribution = Distribution(mean=0.5, sigma=0)
    """Driver imperfection within range [0..1]"""
    min_gap: Distribution = Distribution(mean=2.5, sigma=0)
    """Minimum gap (when standing) in meters."""
    max_speed: float = 55.5
    """The vehicle's maximum velocity (in m/s), defaults to 200 km/h for vehicles"""
    vehicle_type: str = "passenger"
    """The configured vehicle type this actor will perform as. ("passenger", "bus", "coach", "truck", "trailer")"""
    lane_changing_model: LaneChangingModel = field(
        default_factory=LaneChangingModel, hash=False
    )
    junction_model: JunctionModel = field(default_factory=JunctionModel, hash=False)

    def __hash__(self) -> int:
        return pickle_hash_int(self)

    @property
    def id(self) -> str:
        """The identifier tag of the traffic actor."""
        return "{}-{}".format(self.name, str(hash(self))[:6])
