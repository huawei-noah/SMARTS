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
from typing import Optional, Sequence, Set

from smarts.core.actor import ActorRole, ActorState
from smarts.core.controllers import ActionSpaceType
from smarts.core.provider import (
    Provider,
    ProviderManager,
    ProviderRecoveryFlags,
    ProviderState,
)
from smarts.core.road_map import RoadMap
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

    def setup(self, provider_actions) -> ProviderState:
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
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.TargetPose}

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

    def stop_managing(self, actor_id: str):
        pass

    def can_accept_vehicle(self, state: VehicleState) -> bool:
        return False
