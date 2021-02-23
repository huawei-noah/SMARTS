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
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .coordinates import BoundingBox, Pose
from .vehicle import VehicleState


@dataclass
class ProviderTrafficLight:
    lane_in: str
    lane_via: str
    lane_out: str
    state: str


@dataclass
class ProviderTLS:
    tls_id: str
    lights: List[ProviderTrafficLight]


@dataclass
class ProviderState:
    vehicles: List[VehicleState] = field(default_factory=list)
    traffic_light_systems: List[ProviderTLS] = field(default_factory=list)

    def merge(self, other: "ProviderState"):
        our_vehicles = {v.vehicle_id for v in self.vehicles}
        other_vehicles = {v.vehicle_id for v in other.vehicles}
        assert our_vehicles.isdisjoint(other_vehicles)

        our_tlss = {tls.tls_id for tls in self.traffic_light_systems}
        other_tlss = {tls.tls_id for tls in other.traffic_light_systems}
        assert our_tlss.isdisjoint(other_tlss)

        self.vehicles += other.vehicles
        self.traffic_light_systems += other.traffic_light_systems

    def filter(self, vehicle_ids):
        provider_vehicle_ids = [v.vehicle_id for v in self.vehicles]
        for v_id in vehicle_ids:
            try:
                index = provider_vehicle_ids.index(v_id)
                del provider_vehicle_ids[index]
                del self.vehicles[index]
            except ValueError:
                continue
