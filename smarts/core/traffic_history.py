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


from cached_property import cached_property
from contextlib import nullcontext, closing
from functools import lru_cache
import random
import sqlite3


class TrafficHistory:
    def __init__(self, db):
        self._db = db
        self._db_cnxn = None

    def connect_for_multiple_queries(self):
        '''Optional optimization to avoid the overhead of parsing
           the sqlite file header multiple times for clients that
           will be performing multiple queries.  If used, then
           disconnect() should be called when finished.'''
        if not self._db_cnxn:
            self._db_cnxn = sqlite3.connect(self._db)

    def disconnect(self):
        if self._db_cnxn:
            self._db_cnxn.close()
            self._db_cnxn = None

    def _query_val(self, result_type, query, params=()):
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

    def _query_list(self, query, params=()):
        with nullcontext(self._db_cnxn) if self._db_cnxn else closing(
            sqlite3.connect(self._db)
        ) as dbcnxn:
            cur = dbcnxn.cursor()
            for row in cur.execute(query, params):
                yield row
            cur.close()

    @cached_property
    def lane_width(self):
        return self._query_val(float, "SELECT value FROM Spec where key='map_net.lane_width'")

    @cached_property
    def target_speed(self):
        return self._query_val(float, "SELECT value FROM Spec where key='speed_limit_mps'")

    @lru_cache(maxsize=32)
    def vehicle_final_exit_time(self, vehicle_id):
        query = "SELECT max(sim_time) FROM Trajectory WHERE vehicle_id = ?"
        return self._query_val(float, query, params=(vehicle_id,))

    def first_seen_times(self):
        # For now, limit agent missions to just cars (V.type = 2)
        query = """SELECT T.vehicle_id, min(T.sim_time)
            FROM Trajectory AS T INNER JOIN Vehicle AS V ON T.vehicle_id=V.id
            WHERE V.type = 2
            GROUP BY vehicle_id"""
        return self._query_list(query)

    def vehicle_pose_at_time(self, vehicle_id, sim_time):
        query = """SELECT position_x, position_y, heading_rad
                   FROM Trajectory
                   WHERE vehicle_id = ? and sim_time = ?"""
        return self._query_val(
            tuple, query, params=(int(vehicle_id), float(sim_time))
        )

    def vehicle_ids_active_between(self, start_time, end_time):
        query = "SELECT DISTINCT vehicle_id FROM Trajectory WHERE ? <= sim_time AND sim_time <= ?"
        return self._query_list(query, (start_time, end_time))

    def vehicles_active_between(self, start_time, end_time):
        query = """SELECT V.id, V.type, V.length, V.width,
                          T.position_x, T.position_y, T.heading_rad, T.speed
                   FROM Vehicle AS V INNER JOIN Trajectory AS T ON V.id = T.vehicle_id
                   WHERE T.sim_time > ? AND T.sim_time <= ?
                   ORDER BY T.sim_time DESC"""
        return self._query_list(query, (start_time, end_time))

    def random_overlapping_sample(self, vehicle_start_times, k):
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
