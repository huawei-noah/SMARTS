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
from typing import List, Set, Sequence

import numpy as np

from .controllers import ActionSpaceType
from .coordinates import BoundingBox, Pose
from .provider import Provider, ProviderState
from .scenario import Scenario
from .vehicle import VehicleState


class ExternalProvider(Provider):
    def __init__(self, sim):
        self._sim = sim
        self.reset()

    def reset(self):
        self._ext_vehicle_states = []
        self._last_time_delta = None
        self._staleness = None

    def state_update(
        self,
        vehicle_states: Sequence[VehicleState],
        time_delta: float,
        staleness: float,
    ):
        # The bigger 'staleness', the more out of date this state is.
        # Ideally, for better precision, we might set the external state and then step pybullet
        # by an appropriate amount based on staleness, since we can't assume external state
        # is updated in sync with our steps.  However, this is very complicated because we
        # we've *already* simulated this time period.  So we just "eat it" for now!
        assert not time_delta or staleness <= time_delta
        self._ext_vehicle_states = vehicle_states
        self._last_time_delta = time_delta
        self._staleness = staleness

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {}

    def _compute_provider_state(self):
        # TAI: consider deep copying these?
        return ProviderState(vehicles=self._ext_vehicle_states)

    def setup(self, scenario: Scenario) -> ProviderState:
        return self._compute_provider_state()

    def step(self, actions, dt, elapsed_sim_time) -> ProviderState:
        return self._compute_provider_state()

    def sync(self, provider_state: ProviderState):
        pass

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass

    def teardown(self):
        self.reset()

    @property
    def all_vehicle_states(self) -> List[VehicleState]:
        result = []
        for vehicle in self._sim.vehicle_index.vehicles:
            if vehicle.subscribed_to_accelerometer_sensor:
                linear_acc, angular_acc, _, _ = vehicle.accelerometer_sensor(
                    vehicle.state.linear_velocity,
                    vehicle.state.angular_velocity,
                    self._last_time_delta,
                )
            result.append(vehicle.state)
        # TODO: include done, events and reward (if agent)?
        return result
