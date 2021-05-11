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
import logging
import sqlite3
from itertools import cycle
from typing import NamedTuple, Set

from .controllers import ActionSpaceType
from .coordinates import BoundingBox, Heading, Pose
from .provider import Provider, ProviderState
from .vehicle import VEHICLE_CONFIGS, VehicleState


class TrafficHistoryProvider(Provider):
    def __init__(self):
        self._histories_db = None
        self._is_setup = False
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_location_offset = None
        self._replaced_vehicle_ids = set()
        self._start_time_offset = 0

    @property
    def start_time(self):
        return self._start_time_offset

    @start_time.setter
    def start_time(self, start_time: float):
        assert start_time >= 0, "start_time should be positive"
        self._start_time_offset = start_time

    def setup(self, scenario) -> ProviderState:
        if scenario.traffic_history:
            self._histories_db = sqlite3.connect(scenario.traffic_history)
        self._map_location_offset = scenario.road_network.net_offset
        self._is_setup = True
        return ProviderState()

    def set_replaced_ids(self, vehicle_ids: list):
        self._replaced_vehicle_ids.update(vehicle_ids)

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass

    def reset(self):
        pass

    def teardown(self):
        self._is_setup = False
        if self._histories_db:
            self._histories_db.close()
            self._histories_db = None
        self._replaced_vehicle_ids = set()

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {}

    def sync(self, provider_state):
        # Ignore other sim state
        pass

    class _HistoryRow(NamedTuple):
        vehicle_id: int
        vehicle_type: int
        vehicle_length: float
        vehicle_width: float
        position_x: float
        position_y: float
        heading_rad: float
        speed: float

    def _decode_vehicle_type(self, vehicle_type):
        # Options from NGSIM and INTERACTION currently include:
        #  1=motorcycle, 2=auto, 3=truck, 4=pedestrian/bicycle
        if vehicle_type == 1:
            return "motorcycle"
        elif vehicle_type == 2:
            return "passenger"
        elif vehicle_type == 3:
            return "truck"
        elif vehicle_type == 4:
            return "pedestrian"
        else:
            self._log.warning(
                f"unsupported vehicle_type ({vehicle_type}) in history data."
            )
        return "passenger"

    def step(self, provider_actions, dt, elapsed_sim_time) -> ProviderState:
        if not self._histories_db:
            return ProviderState(vehicles=[])
        history_time = self._start_time_offset + elapsed_sim_time
        query = """SELECT V.id, V.type, V.length, V.width,
                          T.position_x, T.position_y, T.heading_rad, T.speed
                   FROM Vehicle AS V INNER JOIN Trajectory AS T ON V.id = T.vehicle_id
                   WHERE T.sim_time > ? AND T.sim_time <= ?
                   ORDER BY T.sim_time DESC"""
        query_args = (history_time - dt, history_time)
        vehicles = []
        vehicle_ids = set()
        cur = self._histories_db.cursor()
        for row in cur.execute(query, query_args):
            hr = TrafficHistoryProvider._HistoryRow(*row)
            v_id = str(hr.vehicle_id)
            if v_id in vehicle_ids or v_id in self._replaced_vehicle_ids:
                continue
            vehicle_ids.add(v_id)
            vehicle_type = self._decode_vehicle_type(hr.vehicle_type)
            default_dims = VEHICLE_CONFIGS[vehicle_type].dimensions
            vehicles.append(
                VehicleState(
                    vehicle_id=f"social-agent-history-{v_id}",
                    vehicle_type=vehicle_type,
                    pose=Pose.from_center(
                        [
                            hr.position_x + self._map_location_offset[0],
                            hr.position_y + self._map_location_offset[1],
                            0,
                        ],
                        Heading(hr.heading_rad),
                    ),
                    dimensions=BoundingBox(
                        length=hr.vehicle_length
                        if hr.vehicle_length is not None
                        else default_dims.length,
                        width=hr.vehicle_width
                        if hr.vehicle_width is not None
                        else default_dims.width,
                        # Note: Neither NGSIM nor INTERACTION provide the vehicle height
                        height=default_dims.height,
                    ),
                    speed=hr.speed,
                    source="HISTORY",
                )
            )
        cur.close()
        return ProviderState(vehicles=vehicles)
