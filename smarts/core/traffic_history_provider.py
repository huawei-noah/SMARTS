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

from typing import Iterable, Optional, Set

from .controllers import ActionSpaceType
from .coordinates import Dimensions, Heading, Pose
from .provider import Provider, ProviderState
from .utils.math import rounder_for_dt
from .vehicle import VEHICLE_CONFIGS, VehicleState


class TrafficHistoryProvider(Provider):
    """A provider that replays traffic history for simulation."""

    def __init__(self):
        self._histories = None
        self._is_setup = False
        self._replaced_vehicle_ids = set()
        self._last_step_vehicles = set()
        self._this_step_dones = set()
        self._vehicle_id_prefix = "history-vehicle-"
        self._start_time_offset = 0

    @property
    def start_time(self):
        """The start time of the traffic playback"""
        return self._start_time_offset

    @start_time.setter
    def start_time(self, start_time: float):
        assert start_time >= 0, "start_time should be positive"
        self._start_time_offset = start_time

    @property
    def done_this_step(self):
        """The vehicles that are to be removed this step."""
        return self._this_step_dones

    def setup(self, scenario) -> ProviderState:
        """Initialize this provider with the given scenario."""
        self._histories = scenario.traffic_history
        if self._histories:
            self._histories.connect_for_multiple_queries()
        self._is_setup = True
        return ProviderState()

    def set_replaced_ids(self, vehicle_ids: Iterable[str]):
        """Replace the given vehicles, excluding them from control by this provider."""
        self._replaced_vehicle_ids.update(vehicle_ids)

    def get_history_id(self, vehicle_id: str) -> Optional[str]:
        """Get the history id of the specified vehicle."""
        if vehicle_id in self._last_step_vehicles:
            return self._vehicle_id_prefix + vehicle_id
        return None

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass

    def reset(self):
        pass

    def teardown(self):
        self._is_setup = False
        if self._histories:
            self._histories.disconnect()
            self._histories = None
        self._replaced_vehicle_ids = set()

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {}

    def sync(self, provider_state):
        # Ignore other sim state
        pass

    def step(
        self, provider_actions, dt: float, elapsed_sim_time: float
    ) -> ProviderState:
        if not self._histories:
            return ProviderState(vehicles=[])
        vehicles = []
        vehicle_ids = set()
        rounder = rounder_for_dt(dt)
        history_time = rounder(self._start_time_offset + elapsed_sim_time)
        prev_time = rounder(history_time - dt)
        rows = self._histories.vehicles_active_between(prev_time, history_time)
        for hr in rows:
            v_id = str(hr.vehicle_id)
            if v_id in vehicle_ids or v_id in self._replaced_vehicle_ids:
                continue
            vehicle_ids.add(v_id)
            vehicle_config_type = self._histories.decode_vehicle_type(hr.vehicle_type)
            vehicles.append(
                VehicleState(
                    vehicle_id=self._vehicle_id_prefix + v_id,
                    vehicle_config_type=vehicle_config_type,
                    pose=Pose.from_center(
                        (hr.position_x, hr.position_y, 0), Heading(hr.heading_rad)
                    ),
                    # Note: Neither NGSIM nor INTERACTION provide the vehicle height
                    dimensions=Dimensions.init_with_defaults(
                        hr.vehicle_length,
                        hr.vehicle_width,
                        hr.vehicle_height,
                        defaults=VEHICLE_CONFIGS[vehicle_config_type].dimensions,
                    ),
                    speed=hr.speed,
                    source="HISTORY",
                )
            )
        self._this_step_dones = {
            self._vehicle_id_prefix + v_id
            for v_id in self._last_step_vehicles - vehicle_ids
        }
        self._last_step_vehicles = vehicle_ids
        return ProviderState(vehicles=vehicles)
