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
from .coordinates import BoundingBox, Pose, Heading
from .provider import ProviderState
from .vehicle import VEHICLE_CONFIGS, VehicleState


class TrafficHistoryProvider:
    def __init__(self):
        self._is_setup = False
        self._current_traffic_history = None
        self.replaced_vehicle_ids = set()

    def setup(self, scenario) -> ProviderState:
        self._is_setup = True
        self._current_traffic_history = scenario.traffic_history
        return ProviderState()

    def set_replaced_ids(self, vehicle_ids: list):
        self.replaced_vehicle_ids.update(vehicle_ids)

    def reset(self):
        pass

    def teardown(self):
        self._is_setup = False
        self._frame = None
        self._current_traffic_history = None
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
                for ts in self._current_traffic_history
                if float(ts) >= elapsed_sim_time
            ),
            default=None,
        )
        if (
            not self._current_traffic_history
            or timestamp is None
            or str(timestamp) not in self._current_traffic_history
        ):
            return ProviderState(vehicles=[], traffic_light_systems=[])

        vehicle_type = "passenger"
        states = ProviderState(
            vehicles=[
                VehicleState(
                    vehicle_id=v_id,
                    vehicle_type=vehicle_type,
                    pose=Pose.from_center(
                        [*vehicle_state["position"][:2], 0,],
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
                for v_id, vehicle_state in self._current_traffic_history[
                    str(timestamp)
                ].items()
                if v_id not in self.replaced_vehicle_ids
            ],
            traffic_light_systems=[],
        )
        return states

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass
