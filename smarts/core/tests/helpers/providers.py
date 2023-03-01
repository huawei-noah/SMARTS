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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Iterable, Optional, Sequence, Set

from smarts.core.actor import ActorRole, ActorState
from smarts.core.controllers import ActionSpaceType
from smarts.core.provider import (
    Provider,
    ProviderManager,
    ProviderRecoveryFlags,
    ProviderState,
)
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.core.traffic_provider import TrafficProvider
from smarts.core.vehicle import VEHICLE_CONFIGS, VehicleState


class MockProvider(Provider):
    def __init__(self):
        self._next_provider_state = None
        self._recovery_flags = super().recovery_flags

    def override_next_provider_state(self, vehicles: Sequence):
        self._next_provider_state = ProviderState(
            actors=[
                VehicleState(
                    actor_id=vehicle_id,
                    vehicle_config_type="passenger",
                    pose=pose,
                    dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
                    speed=speed,
                    source=self.source_str,
                    role=ActorRole.Social,
                )
                for vehicle_id, pose, speed in vehicles
            ],
        )

    def clear_next_provider_state(self):
        self._next_provider_state = None

    def setup(self, scenario: Scenario) -> ProviderState:
        return ProviderState()

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    def set_manager(self, manager: ProviderManager):
        pass

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return set()

    def sync(self, provider_state):
        pass

    def step(self, provider_actions, dt, elapsed_sim_time) -> ProviderState:
        if self._next_provider_state is None:
            return ProviderState(actors=[])

        return self._next_provider_state

    def can_accept_actor(self, state: ActorState) -> bool:
        return True

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional[Provider] = None
    ):
        pass

    def reset(self):
        pass

    def teardown(self):
        self._next_provider_state = None

    def manages_actor(self, actor_id: str) -> bool:
        return True

    @property
    def actor_ids(self) -> Iterable[str]:
        return set(vs.actor_id for vs in self._next_provider_state.actors)

    def stop_managing(self, actor_id: str):
        pass


class MockTrafficProvider(TrafficProvider):
    def __init__(self) -> None:
        self._next_provider_state = None
        self._recovery_flags = super().recovery_flags
        self._ignore = set()

    def override_next_provider_state(self, vehicles: Sequence):
        self._next_provider_state = ProviderState(
            actors=[
                VehicleState(
                    actor_id=vehicle_id,
                    vehicle_config_type="passenger",
                    pose=pose,
                    dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
                    speed=speed,
                    source=self.source_str,
                    role=ActorRole.Social,
                )
                for vehicle_id, pose, speed in vehicles
            ],
        )

    def clear_next_provider_state(self):
        pass

    def setup(self, scenario: Scenario) -> ProviderState:
        return ProviderState()

    def teardown(self):
        self._next_provider_state = None

    def destroy(self):
        pass

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        if self._next_provider_state is None:
            return ProviderState(actors=[])

        return ProviderState(
            actors=[
                a
                for a in self._next_provider_state.actors
                if a.actor_id not in self._ignore
            ]
        )

    def sync(self, provider_state: ProviderState):
        pass

    def reset(self):
        pass

    def set_manager(self, manager: ProviderManager):
        pass

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    def manages_actor(self, actor_id: str) -> bool:
        return actor_id not in self._ignore and any(
            filter(lambda a: a.actor_id == actor_id, self._next_provider_state.actors)
        )

    def can_accept_actor(self, state: ActorState) -> bool:
        return True

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return set()

    def stop_managing(self, actor_id: str):
        self._ignore.add(actor_id)

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional["Provider"] = None
    ):
        self._ignore.discard(provider_actor.actor_id)

    def vehicle_dest_road(self, vehicle_id: str) -> Optional[str]:
        return None

    def update_route_for_vehicle(self, vehicle_id: str, new_route: RoadMap.Route):
        pass

    def vehicle_collided(self, vehicle_id: str):
        pass

    def reserve_traffic_location_for_vehicle(
        self,
        vehicle_id: str,
        reserved_location,
    ):
        pass
