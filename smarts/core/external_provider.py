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
import weakref
from typing import List, Sequence, Set

import numpy as np

from .actor import ActorRole
from .controllers import ActionSpaceType
from .provider import Provider, ProviderManager, ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap
from .scenario import Scenario
from .utils.file import replace
from .vehicle import VehicleState


class ExternalProvider(Provider):
    """A provider that is intended to be used for external intervention in the simulation.
    Vehicles managed by this provider cannot be hijacked by social agents
    and may have privileged VehicleStates."""

    def __init__(self, sim):
        # start with the default recovery flags...
        self._recovery_flags = super().recovery_flags
        self.set_manager(sim)
        self.reset()

    def set_manager(self, manager: ProviderManager):
        self._sim = weakref.ref(manager)

    @property
    def _sim_time(self) -> float:
        sim = self._sim()
        assert sim
        # pytype: disable=attribute-error
        # TAI: consider adding to ProviderManager interface
        return sim.elapsed_sim_time
        # pytype: enable=attribute-error

    @property
    def _vehicle_index(self):
        sim = self._sim()
        assert sim
        # pytype: disable=attribute-error
        # TAI: consider adding to ProviderManager interface
        return sim.vehicle_index
        # pytype: enable=attribute-error

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    def reset(self):
        self._ext_vehicle_states = []
        self._sent_states = None
        self._last_step_delta = None
        self._last_fresh_step = self._sim_time

    def state_update(
        self,
        vehicle_states: Sequence[VehicleState],
        step_delta: float,
    ):
        """Update vehicle states. Use `all_vehicle_states()` to look at previous states."""
        self._ext_vehicle_states = [
            replace(vs, source=self.source_str, role=ActorRole.External)
            for vs in vehicle_states
        ]
        self._last_step_delta = step_delta

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return set()

    @property
    def _provider_state(self):
        dt = self._sim_time - self._last_fresh_step
        if id(self._ext_vehicle_states) != id(self._sent_states):
            self._last_fresh_step = self._sim_time
            self._sent_states = self._ext_vehicle_states
        return ProviderState(actors=self._ext_vehicle_states, dt=dt)

    def setup(self, scenario: Scenario) -> ProviderState:
        return self._provider_state

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        return self._provider_state

    def sync(self, provider_state: ProviderState):
        pass

    def teardown(self):
        self.reset()

    @property
    def all_vehicle_states(self) -> List[VehicleState]:
        """Get all current vehicle states."""
        result = []
        for vehicle in self._vehicle_index.vehicles:
            if vehicle.subscribed_to_accelerometer_sensor:
                linear_acc, angular_acc, _, _ = vehicle.accelerometer_sensor(
                    vehicle.state.linear_velocity,
                    vehicle.state.angular_velocity,
                    self._last_step_delta,
                )
            result.append(vehicle.state)
        return result

    def manages_actor(self, actor_id: str) -> bool:
        for vs in self._ext_vehicle_states:
            if vs.actor_id == actor_id:
                return True
        return False
