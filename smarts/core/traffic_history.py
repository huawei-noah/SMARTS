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

# to allow for typing to refer to class being defined (TrafficHistory)
from __future__ import annotations

import logging
import os
import random
import sqlite3
from contextlib import closing, nullcontext
from functools import lru_cache
from typing import (
    Dict,
    Generator,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from cached_property import cached_property

from smarts.core.coordinates import Dimensions
from smarts.core.utils.math import radians_to_vec
from smarts.core.vehicle import VEHICLE_CONFIGS

T = TypeVar("T")


class TrafficHistory:
    """Traffic history for use with converted datasets."""

    def __init__(self, db):
        self._log = logging.getLogger(self.__class__.__name__)
        self._db = db
        self._db_cnxn = None

    @property
    def name(self) -> str:
        """The name of the traffic history."""
        return os.path.splitext(self._db.name)[0]

    def connect_for_multiple_queries(self):
        """Optional optimization to avoid the overhead of parsing
        the sqlite file header multiple times for clients that
        will be performing multiple queries.  If used, then
        disconnect() should be called when finished."""
        if not self._db_cnxn:
            self._db_cnxn = sqlite3.connect(self._db.path)

    def disconnect(self):
        """End connection with the history database."""
        if self._db_cnxn:
            self._db_cnxn.close()
            self._db_cnxn = None

    def _query_val(
        self, result_type: Type[T], query: str, params: Tuple = ()
    ) -> Optional[T]:
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
    def dataset_source(self) -> str:
        """The known source of the history data"""
        query = "SELECT value FROM Spec where key='source_type'"
        return self._query_val(str, query)

    @cached_property
    def lane_width(self) -> float:
        """The general lane width in the history data"""
        query = "SELECT value FROM Spec where key='map_net.lane_width'"
        return self._query_val(float, query)

    @cached_property
    def target_speed(self) -> float:
        """The general speed limit in the history data."""
        query = "SELECT value FROM Spec where key='speed_limit_mps'"
        return self._query_val(float, query)

    def all_vehicle_ids(self) -> Generator[int, None, None]:
        """Get the ids of all vehicles in the history data"""
        query = "SELECT id FROM Vehicle"
        return (row[0] for row in self._query_list(query))

    @cached_property
    def ego_vehicle_id(self) -> int:
        """The id of the ego's actor in the history data."""
        query = "SELECT id FROM Vehicle WHERE is_ego_vehicle = 1"
        ego_id = self._query_val(int, query)
        return ego_id

    @lru_cache(maxsize=32)
    def vehicle_initial_time(self, vehicle_id: str) -> float:
        """Returns the initial time the specified vehicle is seen in the history data."""
        query = "SELECT MIN(sim_time) FROM Trajectory WHERE vehicle_id = ?"
        return self._query_val(float, query, params=(vehicle_id,))

    @lru_cache(maxsize=32)
    def vehicle_final_exit_time(self, vehicle_id: str) -> float:
        """Returns the final time the specified vehicle is seen in the history data."""
        query = "SELECT MAX(sim_time) FROM Trajectory WHERE vehicle_id = ?"
        return self._query_val(float, query, params=(vehicle_id,))

    @lru_cache(maxsize=32)
    def vehicle_final_position(self, vehicle_id: str) -> Tuple[float, float]:
        """Returns the final (x,y) position for the specified vehicle in the history data."""
        query = "SELECT position_x, position_y FROM Trajectory WHERE vehicle_id=? AND sim_time=(SELECT MAX(sim_time) FROM Trajectory WHERE vehicle_id=?)"
        return self._query_val(
            tuple,
            query,
            params=(
                vehicle_id,
                vehicle_id,
            ),
        )

    def decode_vehicle_type(self, vehicle_type: int) -> str:
        """Convert from the dataset type id to their config type.
        Options from NGSIM and INTERACTION currently include:

        1=motorcycle, 2=auto, 3=truck, 4=pedestrian/bicycle
        This actually returns a ``vehicle_config_type``.
        """
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

    @lru_cache(maxsize=32)
    def vehicle_config_type(self, vehicle_id: str) -> str:
        """Find the configuration type of the specified vehicle."""
        query = "SELECT type FROM Vehicle WHERE id = ?"
        veh_type = self._query_val(int, query, params=(vehicle_id,))
        return self.decode_vehicle_type(veh_type)

    def _resolve_vehicle_dims(
        self, vehicle_type: Union[str, int], length: float, width: float, height: float
    ):
        v_type = vehicle_type
        if isinstance(v_type, int):
            v_type = self.decode_vehicle_type(v_type)
        default_dims = VEHICLE_CONFIGS[v_type].dimensions
        if not length:
            length = default_dims.length
        if not width:
            width = default_dims.width
        if not height:
            height = default_dims.height
        return Dimensions(length, width, height)

    @lru_cache(maxsize=32)
    def vehicle_dims(self, vehicle_id: str) -> Dimensions:
        """Get the vehicle dimensions of the specified vehicle."""
        # do import here to break circular dependency chain
        from smarts.core.vehicle import VEHICLE_CONFIGS

        query = "SELECT length, width, height, type FROM Vehicle WHERE id = ?"
        length, width, height, veh_type = self._query_val(
            tuple, query, params=(vehicle_id,)
        )
        return self._resolve_vehicle_dims(veh_type, length, width, height)

    def first_seen_times(self) -> Generator[Tuple[int, float], None, None]:
        """Find the times each vehicle is first seen in the traffic history.

        XXX: For now, limit agent missions to just passenger cars (V.type = 2)
        """
        query = """SELECT T.vehicle_id, MIN(T.sim_time)
            FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
            WHERE V.type = 2
            GROUP BY vehicle_id"""
        return self._query_list(query)

    def last_seen_vehicle_time(self) -> Optional[float]:
        """Find the time the last vehicle exits the history."""

        query = """SELECT MAX(T.sim_time)
            FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
            WHERE V.type = 2
            ORDER BY T.sim_time DESC LIMIT 1"""
        return self._query_val(float, query)

    def vehicle_pose_at_time(
        self, vehicle_id: str, sim_time: float
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get the pose of the specified vehicle at the specified history time."""
        query = """SELECT position_x, position_y, heading_rad, speed
                   FROM Trajectory
                   WHERE vehicle_id = ? and sim_time = ?"""
        return self._query_val(tuple, query, params=(int(vehicle_id), float(sim_time)))

    def vehicle_ids_active_between(
        self, start_time: float, end_time: float
    ) -> Generator[Tuple, None, None]:
        """Find the ids of all active vehicles between the given history times.

        XXX: For now, limited to just passenger cars (V.type = 2)
        XXX: This looks like the wrong level to filter out vehicles
        """

        query = """SELECT DISTINCT T.vehicle_id
                   FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
                   WHERE ? <= T.sim_time AND T.sim_time <= ? AND V.type = 2"""
        return self._query_list(query, (start_time, end_time))

    class VehicleRow(NamedTuple):
        """Vehicle state information"""

        vehicle_id: int
        vehicle_type: int
        vehicle_length: float
        vehicle_width: float
        vehicle_height: float
        position_x: float
        position_y: float
        heading_rad: float
        speed: float

    class TrafficHistoryVehicleWindow(NamedTuple):
        """General information about a vehicle between a time window."""

        vehicle_id: int
        vehicle_type: int
        vehicle_length: float
        vehicle_width: float
        vehicle_height: float
        start_position_x: float
        start_position_y: float
        start_heading: float
        start_speed: float
        average_speed: float
        start_time: float
        end_time: float
        end_position_x: float
        end_position_y: float
        end_heading: float

        @property
        def axle_start_position(self):
            """The start position of the vehicle from the axle."""
            hhx, hhy = radians_to_vec(self.start_heading) * (0.5 * self.vehicle_length)
            return [self.start_position_x + hhx, self.start_position_y + hhy]

        @property
        def axle_end_position(self):
            """The start position of the vehicle from the axle."""
            hhx, hhy = radians_to_vec(self.end_heading) * (0.5 * self.vehicle_length)
            return [self.end_position_x + hhx, self.end_position_y + hhy]

        @property
        def dimensions(self) -> Dimensions:
            """The known or estimated dimensions of this vehicle."""
            return Dimensions(
                self.vehicle_length, self.vehicle_width, self.vehicle_height
            )

    def vehicles_active_between(
        self, start_time: float, end_time: float
    ) -> Generator[TrafficHistory.VehicleRow, None, None]:
        """Find all vehicles active between the given history times."""
        query = """SELECT V.id, V.type, V.length, V.width, V.height,
                          T.position_x, T.position_y, T.heading_rad, T.speed
                   FROM Vehicle AS V INNER JOIN Trajectory AS T ON V.id = T.vehicle_id
                   WHERE T.sim_time > ? AND T.sim_time <= ?
                   ORDER BY T.sim_time DESC"""
        rows = self._query_list(query, (start_time, end_time))
        return (TrafficHistory.VehicleRow(*row) for row in rows)

    class TrafficLightRow(NamedTuple):
        """Fields in a row from the TrafficLightState table."""

        sim_time: float
        state: int
        stop_point_x: float
        stop_point_y: float
        lane_id: int

    def traffic_light_states_between(
        self, start_time: float, end_time: float
    ) -> Generator[TrafficHistory.TrafficLightRow, None, None]:
        """Find all traffic light states between the given history times."""
        query = """SELECT sim_time, state, stop_point_x, stop_point_y, lane
                   FROM TrafficLightState
                   WHERE sim_time > ? AND sim_time <= ?
                   ORDER BY sim_time ASC"""
        rows = self._query_list(query, (start_time, end_time))
        return (TrafficHistory.TrafficLightRow(*row) for row in rows)

    @staticmethod
    def _greatest_n_per_group_format(
        select, table, group_by, greatest_of_group, where=None, operation="MAX"
    ):
        """This solves the issue where you want to get the highest value of `greatest_of_group` in
        the group `groupby` for versions of sqlite3 that are lower than `3.7.11`.

        See: https://stackoverflow.com/questions/12608025/how-to-construct-a-sqlite-query-to-group-by-order

        e.g. Get a table of the highest speed(`greatest_of_group`) each vehicle(`group_by`) was
        operating at.
        > _greatest_n_per_group_format(
        >    select="vehicle_speed,
        >    vehicle_id",
        >    table="Trajectory",
        >    group_by="vehicle_id",
        >    greatest_of_group="vehicle_speed",
        > )
        """
        where = f"{where} AND" if where else ""
        return f"""
            SELECT {select},
                (SELECT COUNT({group_by}) AS count
                    FROM {table} m2
                    WHERE m1.{group_by} = m2.{group_by})
            FROM  {table} m1
            WHERE {greatest_of_group} = (SELECT {operation}({greatest_of_group})
                                            FROM {table} m3
                                            WHERE {where} m1.{group_by} = m3.{group_by})
            GROUP BY {group_by}
        """

    def _window_from_row(self, row):
        return TrafficHistory.TrafficHistoryVehicleWindow(
            row[0],
            self.decode_vehicle_type(row[1]),
            *self._resolve_vehicle_dims(row[1], *row[2:5]).as_lwh,
            *row[5:],
        )

    def vehicle_window_by_id(
        self,
        vehicle_id: str,
    ) -> Optional[TrafficHistory.TrafficHistoryVehicleWindow]:
        """Find the given vehicle by its id."""
        query = """SELECT V.id, V.type, V.length, V.width, V.height,
                          S.position_x, S.position_y, S.heading_rad, S.speed, D.avg_speed,
                          S.sim_time, E.sim_time,
                          E.position_x, E.position_y, E.heading_rad
                    FROM Vehicle AS V
                    INNER JOIN (
                     SELECT vehicle_id, AVG(speed) as "avg_speed"
                     FROM Trajectory
                     WHERE vehicle_id = ?
                    ) AS D ON V.id = D.vehicle_id
                    INNER JOIN (
                     SELECT vehicle_id, MIN(sim_time) as sim_time, speed, position_x, position_y, heading_rad
                     FROM Trajectory
                     WHERE vehicle_id = ?
                    ) AS S ON V.id = S.vehicle_id
                    INNER JOIN (
                     SELECT vehicle_id, MAX(sim_time) as sim_time, speed, position_x, position_y, heading_rad
                     FROM Trajectory
                     WHERE vehicle_id = ?
                    ) AS E ON V.id = E.vehicle_id
        """
        rows = self._query_list(
            query,
            tuple([vehicle_id] * 3),
        )

        row = next(rows, None)

        if row is None:
            return None
        return self._window_from_row(row)

    def vehicle_windows_in_range(
        self,
        exists_at_or_after: float,
        ends_before: float,
        minimum_vehicle_window: float,
    ) -> Generator[TrafficHistory.TrafficHistoryVehicleWindow, None, None]:
        """Find all vehicles active between the given history times."""
        query = f"""SELECT V.id, V.type, V.length, V.width, V.height,
                          S.position_x, S.position_y, S.heading_rad, S.speed, D.avg_speed,
                          S.sim_time, E.sim_time,
                          E.position_x, E.position_y, E.heading_rad
                   FROM Vehicle AS V
                   INNER JOIN ( 
                    SELECT vehicle_id, AVG(speed) as "avg_speed"
                    FROM (SELECT vehicle_id, sim_time, speed from Trajectory WHERE sim_time >= ? AND sim_time < ?)
                    GROUP BY vehicle_id
                   ) AS D ON V.id = D.vehicle_id
                   INNER JOIN (
                    {self._greatest_n_per_group_format(
                        select='''vehicle_id,
                        sim_time,
                        speed,
                        position_x,
                        position_y,
                        heading_rad''',
                        table='Trajectory',
                        group_by='vehicle_id',
                        greatest_of_group='sim_time',
                        where='sim_time >= ?',
                        operation="MIN"
                    )}
                   ) AS S ON V.id = S.vehicle_id
                   INNER JOIN (
                    {self._greatest_n_per_group_format(
                        select='''vehicle_id,
                        sim_time,
                        speed,
                        position_x,
                        position_y,
                        heading_rad''',
                        table='Trajectory',
                        group_by='vehicle_id',
                        greatest_of_group='sim_time',
                        where='sim_time < ?'
                    )}
                   ) AS E ON V.id = E.vehicle_id
                   WHERE E.sim_time - S.sim_time >= ?
                   GROUP BY V.id
                   ORDER BY S.sim_time
                   """

        rows = self._query_list(
            query,
            (
                exists_at_or_after,
                ends_before,
                exists_at_or_after,
                ends_before,
                minimum_vehicle_window,
            ),
        )

        seen = set()
        seen_pos_x = set()
        rs = list()

        def _window_from_row_debug(row):
            nonlocal seen, seen_pos_x, rs
            r = self._window_from_row(row)
            assert r.vehicle_id not in seen
            assert r.end_time - r.start_time >= minimum_vehicle_window
            assert r.start_time >= exists_at_or_after
            assert r.end_time <= ends_before
            assert r.end_position_x not in seen_pos_x
            seen_pos_x.add(r.end_position_x)
            seen.add(r.vehicle_id)
            rs.append(r)
            return r

        return (_window_from_row_debug(row) for row in rows)

    class TrajectoryRow(NamedTuple):
        """An instant in a trajectory"""

        position_x: float
        position_y: float
        heading_rad: float
        speed: float

    def vehicle_trajectory(
        self, vehicle_id: str
    ) -> Generator[TrafficHistory.TrajectoryRow, None, None]:
        """Get the trajectory of the specified vehicle"""
        query = """SELECT T.position_x, T.position_y, T.heading_rad, T.speed
                   FROM Trajectory AS T
                   WHERE T.vehicle_id = ?"""
        rows = self._query_list(query, (vehicle_id,))
        return (TrafficHistory.TrajectoryRow(*row) for row in rows)

    def random_overlapping_sample(
        self, vehicle_start_times: Dict[str, float], k: int
    ) -> Set[str]:
        """Grab a sample containing a subset of specified vehicles and ensure overlapping time
        intervals across sample.

        Note: this may return a sample with less than k if we're unable to find k overlapping.
        """
        # This is inefficient, but it's not that important
        choice = random.choice(list(vehicle_start_times.keys()))
        sample = {choice}
        sample_start_time = vehicle_start_times[choice]
        sample_end_time = self.vehicle_final_exit_time(choice)
        while len(sample) < k:
            choices = list(vehicle_start_times.keys())
            if len(choices) <= len(sample):
                break
            choice = str(random.choice(choices))
            sample_start_time = min(vehicle_start_times[choice], sample_start_time)
            sample_end_time = max(self.vehicle_final_exit_time(choice), sample_end_time)
            sample.add(choice)
        return sample
