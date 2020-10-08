from typing import List
from dataclasses import dataclass, field

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
