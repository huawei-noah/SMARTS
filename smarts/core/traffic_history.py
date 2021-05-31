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

from __future__ import (
    annotations,
)  # to allow for typing to refer to class being defined (TrafficHistory)
from cached_property import cached_property
from contextlib import nullcontext, closing
from functools import lru_cache
import logging
import os
import random
import sqlite3
from typing import Dict, Generator, NamedTuple, Set, Tuple, TypeVar


class TrafficHistory:
    def __init__(self, db: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._db_cnxn = None

    @property
    def name(self) -> str:
        return os.path.splitext(self._db.name)[0]

    def connect_for_multiple_queries(self):
        """Optional optimization to avoid the overhead of parsing
        the sqlite file header multiple times for clients that
        will be performing multiple queries.  If used, then
        disconnect() should be called when finished."""
        if not self._db_cnxn:
            self._db_cnxn = sqlite3.connect(self._db.path)

    def disconnect(self):
        if self._db_cnxn:
            self._db_cnxn.close()
            self._db_cnxn = None

    def _query_val(
        self, result_type: TypeVar["T"], query: str, params: Tuple = ()
    ) -> T:
        with nullcontext(self._db_cnxn) if self._db_cnxn else closing(
            sqlite3.connect(self._db)
        ) as dbcnxn:
            cur = dbcnxn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            cur.close()
        if not row:
            return None
        return row if result_type is tuple else result_type(row[0])

    def _query_list(
        self, query: str, params: Tuple = ()
    ) -> Generator[Tuple, None, None]:
        with nullcontext(self._db_cnxn) if self._db_cnxn else closing(
            sqlite3.connect(self._db)
        ) as dbcnxn:
            cur = dbcnxn.cursor()
            for row in cur.execute(query, params):
                yield row
            cur.close()

    @cached_property
    def lane_width(self) -> float:
        query = "SELECT value FROM Spec where key='map_net.lane_width'"
        return self._query_val(float, query)

    @cached_property
    def target_speed(self) -> float:
        query = "SELECT value FROM Spec where key='speed_limit_mps'"
        return self._query_val(float, query)

    @lru_cache(maxsize=32)
    def vehicle_final_exit_time(self, vehicle_id: str) -> float:
        query = "SELECT MAX(sim_time) FROM Trajectory WHERE vehicle_id = ?"
        return self._query_val(float, query, params=(vehicle_id,))

    def decode_vehicle_type(self, vehicle_type: int) -> str:
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

    def vehicle_type(self, vehicle_id: str) -> str:
        query = "SELECT type FROM Vehicle WHERE id = ?"
        veh_type = self._query_val(int, query, params=(vehicle_id,))
        return self.decode_vehicle_type(veh_type)

    def vehicle_size(self, vehicle_id: str) -> Tuple[float, float, float]:
        # do import here to break circular dependency chain
        from smarts.core.vehicle import VEHICLE_CONFIGS

        query = "SELECT length, width, type FROM Vehicle WHERE id = ?"
        length, width, veh_type = self._query_val(tuple, query, params=(vehicle_id,))
        default_dims = VEHICLE_CONFIGS[self.decode_vehicle_type(veh_type)].dimensions
        if not length:
            length = default_dims.length
        if not width:
            width = default_dims.width
        # Note: Neither NGSIM nor INTERACTION provide the vehicle height, so use our defaults
        height = default_dims.height
        return length, width, height

    def first_seen_times(self) -> Generator[Tuple[str, float], None, None]:
        # XXX: For now, limit agent missions to just cars (V.type = 2)
        query = """SELECT T.vehicle_id, MIN(T.sim_time)
            FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
            WHERE V.type = 2
            GROUP BY vehicle_id"""
        return self._query_list(query)

    def vehicle_pose_at_time(
        self, vehicle_id: str, sim_time: float
    ) -> Tuple[float, float, float]:
        query = """SELECT position_x, position_y, heading_rad, speed
                   FROM Trajectory
                   WHERE vehicle_id = ? and sim_time = ?"""
        return self._query_val(tuple, query, params=(int(vehicle_id), float(sim_time)))

    def vehicle_ids_active_between(
        self, start_time: float, end_time: float
    ) -> Generator[int, None, None]:
        # XXX: For now, limit agent missions to just cars (V.type = 2)
        query = """SELECT DISTINCT T.vehicle_id
                   FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
                   WHERE ? <= T.sim_time AND T.sim_time <= ? AND V.type = 2"""
        return self._query_list(query, (start_time, end_time))

    class VehicleRow(NamedTuple):
        vehicle_id: int
        vehicle_type: int
        vehicle_length: float
        vehicle_width: float
        position_x: float
        position_y: float
        heading_rad: float
        speed: float

    def vehicles_active_between(
        self, start_time: float, end_time: float
    ) -> Generator[TrafficHistory.VehicleRow, None, None]:
        query = """SELECT V.id, V.type, V.length, V.width,
                          T.position_x, T.position_y, T.heading_rad, T.speed
                   FROM Vehicle AS V INNER JOIN Trajectory AS T ON V.id = T.vehicle_id
                   WHERE T.sim_time > ? AND T.sim_time <= ?
                   ORDER BY T.sim_time DESC"""
        rows = self._query_list(query, (start_time, end_time))
        return (TrafficHistory.VehicleRow(*row) for row in rows)

    def random_overlapping_sample(
        self, vehicle_start_times: Dict[str, float], k: int
    ) -> Set[str]:
        # ensure overlapping time intervals across sample
        # this is inefficient, but it's not that important
        # Note: this may return a sample with less than k
        # if we're unable to find k overlapping.
        choice = random.choice(list(vehicle_start_times.keys()))
        sample = {choice}
        sample_start_time = vehicle_start_times[choice]
        sample_end_time = self.vehicle_final_exit_time(choice)
        while len(sample) < k:
            choices = list(
                self.vehicle_ids_active_between(sample_start_time, sample_end_time)
            )
            if len(choices) <= len(sample):
                break
            choice = str(random.choice(choices)[0])
            sample_start_time = min(vehicle_start_times[choice], sample_start_time)
            sample_end_time = max(self.vehicle_final_exit_time(choice), sample_end_time)
            sample.add(choice)
        return sample
