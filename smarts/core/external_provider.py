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
from typing import List, Sequence, Set

import numpy as np

from .controllers import ActionSpaceType
from .provider import Provider, ProviderState
from .scenario import Scenario
from .vehicle import VehicleState


class ExternalProvider(Provider):
    """A provider that is intended to used for external intervention in the simulation."""

    def __init__(self, sim):
        self._sim = sim
        self.reset()

    def reset(self):
        self._ext_vehicle_states = []
        self._sent_states = None
        self._last_step_delta = None
        self._last_fresh_step = self._sim.elapsed_sim_time

    def state_update(
        self,
        vehicle_states: Sequence[VehicleState],
        step_delta: float,
    ):
        """Update vehicle states. Use `all_vehicle_states()` to look at previous states."""
        self._ext_vehicle_states = vehicle_states
        self._last_step_delta = step_delta

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {}

    @property
    def _provider_state(self):
        dt = self._sim.elapsed_sim_time - self._last_fresh_step
        if id(self._ext_vehicle_states) != id(self._sent_states):
            self._last_fresh_step = self._sim.elapsed_sim_time
            self._sent_states = self._ext_vehicle_states
        return ProviderState(vehicles=self._ext_vehicle_states, dt=dt)

    def setup(self, scenario: Scenario) -> ProviderState:
        return self._provider_state

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        return self._provider_state

    def sync(self, provider_state: ProviderState):
        pass

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass

    def teardown(self):
        self.reset()

    @property
    def all_vehicle_states(self) -> List[VehicleState]:
        """Get all current vehicle states."""
        result = []
        for vehicle in self._sim.vehicle_index.vehicles:
            if vehicle.subscribed_to_accelerometer_sensor:
                linear_acc, angular_acc, _, _ = vehicle.accelerometer_sensor(
                    vehicle.state.linear_velocity,
                    vehicle.state.angular_velocity,
                    self._last_step_delta,
                )
            result.append(vehicle.state)
        return result
