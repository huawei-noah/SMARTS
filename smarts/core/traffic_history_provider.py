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
from itertools import cycle
from typing import Set

from .controllers import ActionSpaceType
from .coordinates import BoundingBox, Heading, Pose
from .provider import ProviderState
from .utils.traffic_history_service import Traffic_history_service
from .vehicle import VEHICLE_CONFIGS, VehicleState


class TrafficHistoryProvider:
    def __init__(self):
        self._is_setup = False
        self._traffic_history_service = None
        self._map_location_offset = None
        self.replaced_vehicle_ids = set()
        self.start_time_offset = 0

    def set_start_time(self, start_time: float):
        assert start_time >= 0, "start_time should be positive"
        self.start_time_offset = start_time

    def setup(self, scenario) -> ProviderState:
        self._is_setup = True
        self._traffic_history_service = scenario._traffic_history_service
        self._map_location_offset = scenario.mapLocationOffset
        return ProviderState()

    def set_replaced_ids(self, vehicle_ids: list):
        self.replaced_vehicle_ids.update(vehicle_ids)

    def reset(self):
        pass

    def teardown(self):
        self._is_setup = False
        self._frame = None
        self._traffic_history_service = None
        self.replaced_vehicle_ids = set()

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {}

    def sync(self, provider_state):
        # Ignore other sim state
        pass

    def step(self, provider_actions, dt, elapsed_sim_time) -> ProviderState:
        timestamp = min(
            (
                float(ts)
                for ts in self._traffic_history_service.all_timesteps
                if float(ts) >= elapsed_sim_time
            ),
            default=None,
        )
        if not self._traffic_history_service or timestamp is None:
            return ProviderState(vehicles=[])

        time_with_offset = str(round(timestamp + self.start_time_offset, 1))
        if not self._traffic_history_service.fetch_history_at_timestep(
            time_with_offset
        ):
            return ProviderState(vehicles=[])

        vehicle_type = "passenger"
        states = ProviderState(
            vehicles=[
                VehicleState(
                    vehicle_id=v_id,
                    vehicle_type=vehicle_type,
                    pose=Pose.from_center(
                        [
                            *Traffic_history_service.apply_map_location_offset(
                                vehicle_state["position"], self._map_location_offset
                            ),
                            0,
                        ],
                        Heading(vehicle_state["heading"]),
                    ),
                    dimensions=BoundingBox(
                        length=vehicle_state.get(
                            "vehicle_length",
                            VEHICLE_CONFIGS[vehicle_type].dimensions.length,
                        ),
                        width=vehicle_state.get(
                            "vehicle_width",
                            VEHICLE_CONFIGS[vehicle_type].dimensions.width,
                        ),
                        # Note: Neither NGSIM nor INTERACTION provide
                        #       the height of the vehicles.
                        height=vehicle_state.get(
                            "vehicle_height",
                            VEHICLE_CONFIGS[vehicle_type].dimensions.height,
                        ),
                    ),
                    speed=vehicle_state["speed"],
                    source="HISTORY",
                )
                for v_id, vehicle_state in self._traffic_history_service.fetch_history_at_timestep(
                    time_with_offset
                ).items()
                if v_id not in self.replaced_vehicle_ids
            ],
        )
        return states

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass
