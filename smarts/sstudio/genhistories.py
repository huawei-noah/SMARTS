# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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


import argparse
import csv
import logging
import math
import os
import sqlite3
import sys
from collections import deque
from typing import Any, Callable, Deque, Dict, Generator, Iterable, Optional

import numpy as np

from smarts.core.coordinates import BoundingBox, Point
from smarts.core.signal_provider import SignalLightState
from smarts.core.utils.file import read_tfrecord_file
from smarts.core.utils.math import (
    circular_mean,
    constrain_angle,
    min_angles_difference_signed,
    vec_to_radians,
)
from smarts.sstudio import types
from smarts.waymo.waymo_utils import WaymoDatasetError

try:
    # pytype: disable=import-error
    from waymo_open_dataset.protos import scenario_pb2
    from waymo_open_dataset.protos.map_pb2 import TrafficSignalLaneState

    # pytype: enable=import-error
except ImportError:
    print(
        "You may not have installed the [waymo] dependencies required to use the waymo replay simulation. Install them first using the command `pip install -e .[waymo]` at the source directory."
    )

METERS_PER_FOOT = 0.3048
DEFAULT_LANE_WIDTH = 3.7  # a typical US highway lane is 12ft ~= 3.7m wide


class _TrajectoryDataset:
    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self.check_dataset_spec(dataset_spec)
        self._output = output
        self._path = os.path.expanduser(dataset_spec["input_path"])
        real_lane_width_m = dataset_spec.get("real_lane_width_m", DEFAULT_LANE_WIDTH)
        lane_width = dataset_spec.get("map_lane_width", real_lane_width_m)
        self._scale = lane_width / real_lane_width_m
        self._flip_y = dataset_spec.get("flip_y", False)
        self._swap_xy = dataset_spec.get("swap_xy", False)
        # most trajectory datasets have .1s time delta (i.e., were collected at 10 Hz)
        self._dt_sec = 0.1

    class _WindowedReader:
        """Iterates over the rows in file using a sliding window that keeps track of
        both a number of rows before the current row and a number of rows after.
        These "windows" are passed to a row transformation function on each step.
        For example, if window_before = 4 and window_after = 3, for a 9-row
        file, the windows associated with each row are:
            row 1, before = [], after = [2, 3, 4]
            row 2, before = [1], after = [3, 4, 5]
            row 3, before = [2, 1], after = [4, 5, 6]
            row 4, before = [3, 2, 1], after = [5, 6, 7]
            row 5, before = [4, 3, 2, 1], after = [6, 7, 8]
            row 6, before = [5, 4, 3, 2], after = [7, 8, 9]
            row 7, before = [6, 5, 4, 3], after = [8, 9]
            row 8, before = [7, 6, 5, 4], after = [9]
            row 9, before = [8, 7, 6, 5], after = []
        Windows are cleared whenever the value in the (optional) `group_col` column changes.
        This was designed to be nestable by making the `row_gen` parameter support an iterator over another _WindowedReader.
        """

        Row = Dict[str, Any]

        def __init__(
            self,
            row_gen: Iterable[Row],
            transform_fn: Callable[[Row, Deque[Row], Deque[Row]], None],
            window_before: int = 0,
            window_after: int = 0,
            group_col: Optional[str] = None,
        ):
            self._row_gen = row_gen
            self._transform_fn = transform_fn
            self._before_width = window_before
            self._after_width = window_after
            self._group_col = group_col

        def __iter__(self) -> Generator[Row, None, None]:
            after_win = deque(maxlen=self._after_width)
            before_win = deque(maxlen=self._before_width)
            cur_row = None
            prev_group = None
            for row in self._row_gen:
                if self._group_col and row[self._group_col] != prev_group:
                    while after_win:
                        if cur_row:
                            before_win.appendleft(cur_row)
                        cur_row = after_win.popleft()
                        self._transform_fn(cur_row, before_win, after_win)
                        yield cur_row
                    before_win.clear()
                    cur_row = None
                if self._group_col:
                    prev_group = row[self._group_col]
                if len(after_win) < self._after_width:
                    after_win.append(row)
                    continue
                if cur_row:
                    before_win.appendleft(cur_row)
                cur_row = after_win.popleft() if after_win else row
                after_win.append(row)
                self._transform_fn(cur_row, before_win, after_win)
                yield cur_row
            while after_win:
                if cur_row:
                    before_win.appendleft(cur_row)
                cur_row = after_win.popleft()
                self._transform_fn(cur_row, before_win, after_win)
                yield cur_row

    @property
    def scale(self) -> float:
        """The base scale based on the ratio of map lane size to real lane size."""
        return self._scale

    @property
    def traffic_light_rows(self) -> Iterable:
        """Iterable dataset rows representing traffic light states (if present)."""
        raise NotImplementedError

    @property
    def rows(self) -> Iterable:
        """The iterable rows of the dataset."""
        raise NotImplementedError

    def column_val_in_row(self, row, col_name: str) -> Any:
        """Access the value of a dataset row which intersects with the given column name."""
        # XXX: this public method is improper because this requires a dataset row but that is
        # implementation specific.
        raise NotImplementedError

    def check_dataset_spec(self, dataset_spec: Dict[str, Any]):
        """Validate the form of the dataset specification."""
        errmsg = None
        if "input_path" not in dataset_spec:
            errmsg = "'input_path' field is required in dataset_spec."
        elif dataset_spec.get("flip_y"):
            if dataset_spec["source_type"] != "NGSIM":
                errmsg = "'flip_y' option only supported for NGSIM datasets."
            elif not dataset_spec.get("_map_bbox"):
                errmsg = "'_map_bbox' is required if 'flip_y' option used; need to pass in a map_spec."
        if errmsg:
            self._log.error(errmsg)
            raise ValueError(errmsg)
        self._dataset_spec = dataset_spec

    def _write_dict(self, curdict: Dict, insert_sql: str, cursor, curkey: str = ""):
        for key, value in curdict.items():
            newkey = f"{curkey}.{key}" if curkey else key
            if isinstance(value, dict):
                self._write_dict(value, insert_sql, cursor, newkey)
            else:
                cursor.execute(insert_sql, (newkey, str(value)))

    def _create_tables(self, dbconxn):
        ccur = dbconxn.cursor()
        ccur.execute(
            """CREATE TABLE Spec (
                   key TEXT PRIMARY KEY,
                   value TEXT
               ) WITHOUT ROWID"""
        )
        ccur.execute(
            """CREATE TABLE Vehicle (
                   id INTEGER PRIMARY KEY,
                   type INTEGER NOT NULL,
                   length REAL,
                   width REAL,
                   height REAL,
                   is_ego_vehicle INTEGER DEFAULT 0
               ) WITHOUT ROWID"""
        )
        ccur.execute(
            """CREATE TABLE Trajectory (
                   vehicle_id INTEGER NOT NULL,
                   sim_time REAL NOT NULL,
                   position_x REAL NOT NULL,
                   position_y REAL NOT NULL,
                   heading_rad REAL NOT NULL,
                   speed REAL DEFAULT 0.0,
                   lane_id INTEGER DEFAULT 0,
                   PRIMARY KEY (vehicle_id, sim_time),
                   FOREIGN KEY (vehicle_id) REFERENCES Vehicle(id)
               ) WITHOUT ROWID"""
        )
        ccur.execute(
            """CREATE TABLE TrafficLightState (
                   sim_time REAL NOT NULL,
                   state INTEGER NOT NULL,
                   stop_point_x REAL NOT NULL,
                   stop_point_y REAL NOT NULL,
                   lane INTEGER NOT NULL
               )"""
        )
        dbconxn.commit()
        ccur.close()

    def create_output(self, time_precision: int = 3):
        """Convert the dataset into the output database file.

        Args:
            time_precision: A limit for digits after decimal for each processed sim_time.
                (3 is millisecond precision)
        """
        dbconxn = sqlite3.connect(self._output)

        self._log.debug("creating tables...")
        self._create_tables(dbconxn)

        self._log.debug("inserting data...")

        iscur = dbconxn.cursor()
        insert_kv_sql = "INSERT INTO Spec VALUES (?, ?)"
        self._write_dict(self._dataset_spec, insert_kv_sql, iscur)
        dbconxn.commit()
        iscur.close()

        # TAI:  can use executemany() and batch insert rows together if this turns out to be too slow...
        insert_vehicle_sql = "INSERT INTO Vehicle VALUES (?, ?, ?, ?, ?, ?)"
        insert_traj_sql = "INSERT INTO Trajectory VALUES (?, ?, ?, ?, ?, ?, ?)"
        insert_traffic_light_sql = (
            "INSERT INTO TrafficLightState VALUES (?, ?, ?, ?, ?)"
        )
        vehicle_ids = set()
        itcur = dbconxn.cursor()

        x_offset = self._dataset_spec.get("x_offset", 0.0)
        y_offset = self._dataset_spec.get("y_offset", 0.0)
        for row in self.rows:
            vid = int(self.column_val_in_row(row, "vehicle_id"))
            if vid not in vehicle_ids:
                ivcur = dbconxn.cursor()

                # These are not available in all datasets
                height = self.column_val_in_row(row, "height")
                is_ego = self.column_val_in_row(row, "is_ego_vehicle")

                veh_args = (
                    vid,
                    int(self.column_val_in_row(row, "type")),
                    float(self.column_val_in_row(row, "length")) * self.scale,
                    float(self.column_val_in_row(row, "width")) * self.scale,
                    float(height) * self.scale if height else None,
                    int(is_ego) if is_ego else 0,
                )
                ivcur.execute(insert_vehicle_sql, veh_args)
                ivcur.close()
                dbconxn.commit()
                vehicle_ids.add(vid)
            traj_args = (
                vid,
                # time units are in milliseconds for both NGSIM and Interaction datasets, convert to secs
                round(
                    float(self.column_val_in_row(row, "sim_time")) / 1000,
                    time_precision,
                ),
                (float(self.column_val_in_row(row, "position_x")) + x_offset)
                * self.scale,
                (float(self.column_val_in_row(row, "position_y")) + y_offset)
                * self.scale,
                float(self.column_val_in_row(row, "heading_rad")),
                float(self.column_val_in_row(row, "speed")) * self.scale,
                self.column_val_in_row(row, "lane_id"),
            )
            # Ignore datapoints with NaNs
            if not any(a is not None and np.isnan(a) for a in traj_args):
                itcur.execute(insert_traj_sql, traj_args)

        # Insert traffic light states if available
        try:
            for row in self.traffic_light_rows:
                tls_args = (
                    round(
                        float(self.column_val_in_row(row, "sim_time")) / 1000,
                        time_precision,
                    ),
                    int(self.column_val_in_row(row, "state")),
                    float(self.column_val_in_row(row, "stop_point_x") + x_offset)
                    * self.scale,
                    float(self.column_val_in_row(row, "stop_point_y") + y_offset)
                    * self.scale,
                    float(self.column_val_in_row(row, "lane")),
                )
                itcur.execute(insert_traffic_light_sql, tls_args)
        except NotImplementedError:
            pass

        itcur.close()
        dbconxn.commit()

        # ensure that sim_time always starts at 0:
        self._log.debug("shifting sim_times..")
        mcur = dbconxn.cursor()
        mcur.execute(
            f"UPDATE Trajectory SET sim_time = round(sim_time - (SELECT min(sim_time) FROM Trajectory), {time_precision})"
        )
        mcur.close()
        dbconxn.commit()

        self._log.debug("creating indices..")
        icur = dbconxn.cursor()
        icur.execute("CREATE INDEX Trajectory_Time ON Trajectory (sim_time)")
        icur.execute("CREATE INDEX Trajectory_Vehicle ON Trajectory (vehicle_id)")
        icur.execute("CREATE INDEX Vehicle_Type ON Vehicle (type)")
        icur.execute(
            "CREATE INDEX TrafficLightState_SimTime ON TrafficLightState (sim_time)"
        )
        dbconxn.commit()
        icur.close()

        dbconxn.close()
        self._log.debug("output done")


class Interaction(_TrajectoryDataset):
    """A tool to convert a dataset to a database for use in SMARTS."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        super().__init__(dataset_spec, output)
        assert not self._flip_y
        self._max_angular_velocity = dataset_spec.get("max_angular_velocity", None)
        self._heading_min_speed = dataset_spec.get("heading_inference_min_speed", 2.2)
        self._prev_heading = None
        self._next_row = None

    def check_dataset_spec(self, dataset_spec: Dict[str, Any]):
        super().check_dataset_spec(dataset_spec)
        hiw = dataset_spec.get("heading_inference_window", 2)
        # 11 is a semi-arbitrary max just to keep things "sane".
        if not 2 <= hiw <= 11:
            raise ValueError("heading_inference_window must be between 2 and 11")

    def _lookup_agent_type(self, agent_type: str) -> int:
        # Try to match the NGSIM types...
        if agent_type == "motorcycle":
            return 1
        elif agent_type == "car":
            return 2
        elif agent_type == "truck":
            return 3
        elif agent_type == "pedestrian/bicycle":
            return 4
        self._log.warning(f"unknown agent_type:  {agent_type}.")
        return 0

    def _row_gen(self) -> Generator[_TrajectoryDataset._WindowedReader.Row, None, None]:
        x_margin = self._dataset_spec.get("x_margin_px", 0) / self.scale
        y_margin = self._dataset_spec.get("y_margin_px", 0) / self.scale
        with open(self._path, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                # See: https://interaction-dataset.com/details-and-format
                # position and length/width are in meters.
                # Note: track_id will be like "P12" for pedestrian tracks.  (TODO)
                row["vehicle_id"] = int(row["track_id"])
                row["sim_time"] = row["timestamp_ms"]
                if self._swap_xy:
                    row["position_x"] = float(row["y"])
                    row["position_y"] = float(row["x"])
                    row["vx"] = float(row["vy"])
                    row["vy"] = float(row["vx"])
                else:
                    row["position_x"] = float(row["x"])
                    row["position_y"] = float(row["y"])
                    row["vx"] = float(row["vx"])
                    row["vy"] = float(row["vy"])
                row["length"] = float(row.get("length", 0.0))
                row["width"] = float(row.get("width", 0.0))
                row["type"] = self._lookup_agent_type(row["agent_type"])

                # offset of the map from the data...
                if x_margin:
                    row["position_x"] -= x_margin
                if y_margin:
                    row["position_y"] -= y_margin
                if self._flip_y:
                    map_bb = self._dataset_spec["_map_bbox"]
                    row["position_y"] = map_bb.max_pt.y / self.scale - row["position_y"]

                yield row

    def _cal_speed(
        self,
        row: _TrajectoryDataset._WindowedReader.Row,
        before_win: Deque[_TrajectoryDataset._WindowedReader.Row],
        after_win: Deque[_TrajectoryDataset._WindowedReader.Row],
    ):
        row["speed"] = np.linalg.norm((row["vx"], row["vy"]))
        if after_win:
            c = np.array((row["position_x"], row["position_y"]))
            n = np.array((after_win[0]["position_x"], after_win[0]["position_y"]))
            if not any(np.isnan(c)) and not any(np.isnan(n)):
                # XXX: could try to divide by sim_time delta here instead of assuming it's fixed
                row["speed"] = np.linalg.norm(n - c) / self._dt_sec

    def _infer_heading(
        self,
        row: _TrajectoryDataset._WindowedReader.Row,
        before_win: Deque[_TrajectoryDataset._WindowedReader.Row],
        after_win: Deque[_TrajectoryDataset._WindowedReader.Row],
    ):
        window = [np.array((r["position_x"], r["position_y"])) for r in before_win]
        window.reverse()
        window += [np.array((row["position_x"], row["position_y"]))]
        window += [np.array((r["position_x"], r["position_y"])) for r in after_win]
        speeds = (
            [r["speed"] for r in before_win]
            + [row["speed"]]
            + [r["speed"] for r in after_win]
        )
        vecs = []
        prev_vhat = None
        prev_inst_heading = None
        for w in range(len(window) - 1):
            c = window[w]
            n = window[w + 1]
            if any(np.isnan(c)) or any(np.isnan(n)):
                if prev_vhat is not None:
                    vecs.append(prev_vhat)
                continue
            s = np.linalg.norm(n - c)
            if s == 0.0 or (
                self._heading_min_speed is not None
                and (
                    (s / self._dt_sec) < self._heading_min_speed
                    or speeds[w] < self._heading_min_speed
                )
            ):
                if prev_vhat is not None:
                    vecs.append(prev_vhat)
                continue
            vhat = (n - c) / s
            inst_heading = vec_to_radians(vhat)
            if prev_inst_heading is not None:
                if self._max_angular_velocity:
                    # XXX: could try to divide by sim_time delta here instead of assuming it's fixed
                    angular_velocity = (
                        min_angles_difference_signed(inst_heading, prev_inst_heading)
                        / self._dt_sec
                    )
                    if abs(angular_velocity) > self._max_angular_velocity:
                        inst_heading = (
                            prev_inst_heading
                            + np.sign(angular_velocity)
                            * self._max_angular_velocity
                            * self._dt_sec
                        )
                        inst_heading += 0.5 * math.pi
                        vhat = np.array(
                            (math.cos(inst_heading), math.sin(inst_heading))
                        )
            vecs.append(vhat)
            prev_vhat = vhat
            prev_inst_heading = inst_heading
        if vecs:
            new_heading = circular_mean(vecs)
        elif self._prev_heading is not None:
            new_heading = self._prev_heading
        elif "psi_rad" in row:
            new_heading = float(row["psi_rad"]) - 0.5 * math.pi
        else:
            new_heading = self._default_heading
        self._prev_heading = new_heading
        row["heading_rad"] = new_heading % (2 * math.pi)

    @property
    def rows(self) -> Generator[Dict, None, None]:
        self._log.debug("transforming Interaction data...")

        # first calculate speeds based on positions (instead of vx, vy)
        # since dataset speeds are "instantaneous"and so don't match with dPos/dt, which can affect some models.
        speeds_gen = _TrajectoryDataset._WindowedReader(
            self._row_gen(), self._cal_speed, 0, 1, "vehicle_id"
        )

        # now infer heading with rolling window...
        heading_window = self._dataset_spec.get("heading_inference_window", 2)
        heading_before_win = int((heading_window / 2) + (heading_window % 2) - 1)
        heading_after_win = int(heading_window / 2)
        headings_gen = _TrajectoryDataset._WindowedReader(
            speeds_gen,
            self._infer_heading,
            heading_before_win,
            heading_after_win,
            "vehicle_id",
        )

        map_bbox = self._dataset_spec.get("_map_bbox")

        # note: iterating over outer generator iterates over all nested generators too...
        # XXX: assumes all timesteps for a vehicle are grouped together in the file and are in sorted temporal order
        for row in headings_gen:
            if map_bbox and not map_bbox.contains(
                Point(self.scale * row["position_x"], self.scale * row["position_y"])
            ):
                self._log.info(
                    f"skipping row for vehicle {row['vehicle_id']} with position off of map"
                )
                continue
            yield row

    def column_val_in_row(self, row, col_name: str) -> Any:
        return row.get(col_name)


class NGSIM(_TrajectoryDataset):
    """A tool for conversion of a NGSIM dataset for use within SMARTS."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        super().__init__(dataset_spec, output)
        # self._prev_heading = 3 * math.pi / 2
        self._prev_heading = None
        self._default_heading = dataset_spec.get("default_heading", 3.0 * math.pi / 2.0)
        self._max_angular_velocity = dataset_spec.get("max_angular_velocity", None)
        # 2.2 corresponds to roughly 5mph.
        self._heading_min_speed = dataset_spec.get("heading_inference_min_speed", 2.2)
        self._determine_columns()

    def check_dataset_spec(self, dataset_spec: Dict[str, Any]):
        super().check_dataset_spec(dataset_spec)
        hiw = dataset_spec.get("heading_inference_window", 2)
        # 11 is a semi-arbitrary max just to keep things "sane".
        if not 2 <= hiw <= 11:
            raise ValueError("heading_inference_window must be between 2 and 11")

    def _determine_columns(self):
        self._columns = (
            "vehicle_id",
            "frame_id",  # 1 frame per .1s
            "total_frames",
            "sim_time",  # msecs
            # front center in feet from left lane edge
            "position_x" if not self._swap_xy else "position_y",
            # front center in feet from entry edge
            "position_y" if not self._swap_xy else "position_x",
            "global_x" if not self._swap_xy else "global_y",  # front center in feet
            "global_y" if not self._swap_xy else "global_x",  # front center in feet
            "length",  # feet
            "width",  # feet
            "type",  # 1 = motorcycle, 2 = auto, 3 = truck
            "speed",  # feet / sec
            "acceleration",  # feet / sec^2
            "lane_id",  # lower is further left
            "preceding_vehicle_id",
            "following_vehicle_id",
            "spacing",  # feet
            "headway",  # secs
        )
        with open(self._path, newline="") as infile:
            num_cols = len(infile.readline().strip().split())
        if num_cols > len(self._columns):
            extra_cols = (
                "origin_zone",
                "destination_zone",
                "intersection",
                "section",
                "direction",
                "movement",
            )
            self._columns = self._columns[:16] + extra_cols + self._columns[16:]
        assert num_cols == len(
            self._columns
        ), f"unexpected number of columns/fields ({num_cols}) in {self._path}"

    def _smooth_positions(
        self,
        row: _TrajectoryDataset._WindowedReader.Row,
        before_win: Deque[_TrajectoryDataset._WindowedReader.Row],
        after_win: Deque[_TrajectoryDataset._WindowedReader.Row],
    ):
        pos_width = 1 + before_win.maxlen + after_win.maxlen
        sumwin = (
            lambda d, key: sum(
                d[r][key] if r < len(d) else d[-1][key] for r in range(d.maxlen)
            )
            if d
            else row[key] * d.maxlen
        )
        row["position_x"] += sumwin(before_win, "position_x") + sumwin(
            after_win, "position_x"
        )
        row["position_x"] /= pos_width
        row["position_y"] += sumwin(before_win, "position_y") + sumwin(
            after_win, "position_y"
        )
        row["position_y"] /= pos_width

    def _infer_heading(
        self,
        row: _TrajectoryDataset._WindowedReader.Row,
        before_win: Deque[_TrajectoryDataset._WindowedReader.Row],
        after_win: Deque[_TrajectoryDataset._WindowedReader.Row],
    ):
        window = [np.array((r["position_x"], r["position_y"])) for r in before_win]
        window.reverse()
        window += [np.array((row["position_x"], row["position_y"]))]
        window += [np.array((r["position_x"], r["position_y"])) for r in after_win]
        speeds = (
            [r["speed"] for r in before_win]
            + [row["speed"]]
            + [r["speed"] for r in after_win]
        )
        vecs = []
        prev_vhat = None
        prev_inst_heading = None
        for w in range(len(window) - 1):
            c = window[w]
            n = window[w + 1]
            if any(np.isnan(c)) or any(np.isnan(n)):
                if prev_vhat is not None:
                    vecs.append(prev_vhat)
                continue
            s = np.linalg.norm(n - c)
            if s == 0.0 or (
                self._heading_min_speed is not None
                and (
                    (s / self._dt_sec) < self._heading_min_speed
                    or speeds[w] < self._heading_min_speed
                )
            ):
                if prev_vhat is not None:
                    vecs.append(prev_vhat)
                continue
            vhat = (n - c) / s
            inst_heading = vec_to_radians(vhat)
            if prev_inst_heading is not None:
                if self._max_angular_velocity:
                    # XXX: could try to divide by sim_time delta here instead of assuming it's fixed
                    angular_velocity = (
                        min_angles_difference_signed(inst_heading, prev_inst_heading)
                        / self._dt_sec
                    )
                    if abs(angular_velocity) > self._max_angular_velocity:
                        inst_heading = (
                            prev_inst_heading
                            + np.sign(angular_velocity)
                            * self._max_angular_velocity
                            * self._dt_sec
                        )
                        inst_heading += 0.5 * math.pi
                        vhat = np.array(
                            (math.cos(inst_heading), math.sin(inst_heading))
                        )
            vecs.append(vhat)
            prev_vhat = vhat
            prev_inst_heading = inst_heading
        if vecs:
            new_heading = circular_mean(vecs)
        elif self._prev_heading is None:
            # TAI:  backfill from the first "real" heading (second pass)
            new_heading = self._default_heading
        else:
            new_heading = self._prev_heading
        self._prev_heading = new_heading
        row["heading_rad"] = new_heading % (2 * math.pi)

        # now since SMARTS' positions are the vehicle centerpoints, but NGSIM's are at the front
        # we must adjust the vehicle position to its centerpoint based on its inferred heading angle (+y = 0 rad)
        adj_heading = row["heading_rad"] + 0.5 * math.pi
        half_len = 0.5 * row["length"]
        # XXX: need to use a different key heree since changing position_x or position_y would probably
        # XXX: affect a row that's still in the before window of a nested generator (smooth_positions).
        row["adj_position_x"] = row["position_x"] - half_len * np.cos(adj_heading)
        row["adj_position_y"] = row["position_y"] - half_len * np.sin(adj_heading)

    def _cal_speed(
        self,
        row: _TrajectoryDataset._WindowedReader.Row,
        before_win: Deque[_TrajectoryDataset._WindowedReader.Row],
        after_win: Deque[_TrajectoryDataset._WindowedReader.Row],
    ):
        row["speed_discrete"] = None
        if not after_win:
            return row
        c = np.array((row["adj_position_x"], row["adj_position_y"]))
        n = np.array((after_win[0]["adj_position_x"], after_win[0]["adj_position_y"]))
        if not any(np.isnan(c)) and not any(np.isnan(n)):
            # XXX: could try to divide by sim_time delta here instead of assuming it's fixed
            row["speed_discrete"] = np.linalg.norm(n - c) / self._dt_sec

    def _row_gen(self) -> Generator[_TrajectoryDataset._WindowedReader.Row, None, None]:
        x_margin = self._dataset_spec.get("x_margin_px", 0) / self.scale
        y_margin = self._dataset_spec.get("y_margin_px", 0) / self.scale
        with open(self._path, newline="") as infile:
            for line in infile:
                fields = line.split()
                row = {col: fields[f] for f, col in enumerate(self._columns)}

                row["lane_id"] = int(row["lane_id"])
                row["length"] = float(row["length"]) * METERS_PER_FOOT
                row["width"] = float(row["width"]) * METERS_PER_FOOT
                row["speed"] = float(row["speed"]) * METERS_PER_FOOT
                row["acceleration"] = float(row["acceleration"]) * METERS_PER_FOOT
                row["spacing"] = float(row["spacing"]) * METERS_PER_FOOT
                row["position_x"] = float(row["position_x"]) * METERS_PER_FOOT
                row["position_y"] = float(row["position_y"]) * METERS_PER_FOOT

                # offset of the map from the data...
                if x_margin:
                    row["position_x"] -= x_margin
                if y_margin:
                    row["position_y"] -= y_margin
                if self._flip_y:
                    map_bb = self._dataset_spec["_map_bbox"]
                    row["position_y"] = map_bb.max_pt.y / self.scale - row["position_y"]

                yield row

    @property
    def rows(self) -> Generator[Dict, None, None]:
        self._log.debug("transforming NGSIM data...")

        # smooth positions using a moving average...
        # TAI: make this window size a parameter too?
        posns_gen = _TrajectoryDataset._WindowedReader(
            self._row_gen(), self._smooth_positions, 7, 7, "vehicle_id"
        )

        # infer heading with rolling window on previously-smoothed positions...
        heading_window = self._dataset_spec.get("heading_inference_window", 2)
        heading_before_win = int((heading_window / 2) + (heading_window % 2) - 1)
        heading_after_win = int(heading_window / 2)
        headings_gen = _TrajectoryDataset._WindowedReader(
            posns_gen,
            self._infer_heading,
            heading_before_win,
            heading_after_win,
            "vehicle_id",
        )

        # finally calculate speeds based on these smoothed and centered positions...
        # (This also overcomes problem that NGSIM speeds are "instantaneous"
        # and so don't match with dPos/dt, which can affect some models.)
        speeds_gen = _TrajectoryDataset._WindowedReader(
            headings_gen, self._cal_speed, 0, 1, "vehicle_id"
        )

        map_bbox = self._dataset_spec.get("_map_bbox")

        # note: iterating over outer generator iterates over all nested generators too...
        # XXX: assumes all timesteps for a vehicle are grouped together in the file and are in sorted temporal order
        for row in speeds_gen:
            if map_bbox and not map_bbox.contains(
                Point(
                    self.scale * row["adj_position_x"],
                    self.scale * row["adj_position_y"],
                )
            ):
                self._log.info(
                    f"skipping row for vehicle {row['vehicle_id']} with position off of map"
                )
                continue
            yield row

    def column_val_in_row(self, row, col_name: str) -> Any:
        if col_name == "speed":
            return row["speed_discrete"] if row["speed_discrete"] else row["speed"]
        if col_name == "position_x":
            return row["adj_position_x"]
        if col_name == "position_y":
            return row["adj_position_y"]
        return row.get(col_name)


class Waymo(_TrajectoryDataset):
    """A tool for conversion of a Waymo dataset for use within SMARTS."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        super().__init__(dataset_spec, output)

    def _get_scenario(self):
        if "scenario_id" not in self._dataset_spec:
            errmsg = "Dataset spec requires scenario_id to be set"
            self._log.error(errmsg)
            raise ValueError(errmsg)
        scenario_id = self._dataset_spec["scenario_id"]
        # Loop over the scenarios in the TFRecord and check its ID for a match
        scenario = None
        dataset = read_tfrecord_file(self._dataset_spec["input_path"])
        for record in dataset:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                return parsed_scenario
        raise ValueError(
            f"Dataset file does not contain scenario with id: {scenario_id}"
        )

    @property
    def rows(self) -> Generator[Dict, None, None]:
        def lerp(a: float, b: float, t: float) -> float:
            return t * (b - a) + a

        scenario = self._get_scenario()

        for i in range(len(scenario.tracks)):
            vehicle_id = scenario.tracks[i].id
            vehicle_type = self._lookup_agent_type(scenario.tracks[i].object_type)
            num_steps = len(scenario.timestamps_seconds)
            rows = []

            # First pass -- extract data
            for j in range(num_steps):
                obj_state = scenario.tracks[i].states[j]
                vel = np.array([obj_state.velocity_x, obj_state.velocity_y])

                row = dict()
                row["valid"] = obj_state.valid
                row["vehicle_id"] = vehicle_id
                row["type"] = vehicle_type
                row["length"] = obj_state.length
                row["height"] = obj_state.height
                row["width"] = obj_state.width
                row["sim_time"] = scenario.timestamps_seconds[j]
                row["position_x"] = obj_state.center_x
                row["position_y"] = obj_state.center_y
                row["heading_rad"] = (obj_state.heading - math.pi / 2) % (2 * math.pi)
                row["speed"] = np.linalg.norm(vel)
                row["lane_id"] = 0
                row["is_ego_vehicle"] = 1 if i == scenario.sdc_track_index else 0
                rows.append(row)

            # Second pass -- align timesteps to 10 Hz and interpolate trajectory data if needed
            interp_rows = [None] * num_steps
            for j in range(num_steps):
                row = rows[j]
                time_current = row["sim_time"]
                time_expected = round(j * self._dt_sec, 3)
                time_error = time_current - time_expected

                if round(abs(time_error), 1) >= self._dt_sec:
                    raise WaymoDatasetError(
                        f"[{scenario.scenario_id}] Waymo data deviates by more than the size of 1 timestep. This likely indicates a gap in the dataset."
                    )

                if not row["valid"] or time_error == 0:
                    continue

                if time_error > 0:
                    # We can't interpolate if the previous element doesn't exist or is invalid
                    if j == 0 or not rows[j - 1]["valid"]:
                        continue

                    # Interpolate backwards using previous timestep
                    interp_row = {"sim_time": time_expected}

                    prev_row = rows[j - 1]
                    prev_time = prev_row["sim_time"]

                    t = (time_expected - prev_time) / (time_current - prev_time)
                    interp_row["speed"] = lerp(prev_row["speed"], row["speed"], t)
                    interp_row["position_x"] = lerp(
                        prev_row["position_x"], row["position_x"], t
                    )
                    interp_row["position_y"] = lerp(
                        prev_row["position_y"], row["position_y"], t
                    )
                    interp_row["heading_rad"] = lerp(
                        prev_row["heading_rad"], row["heading_rad"], t
                    )
                    interp_rows[j] = interp_row
                else:
                    # We can't interpolate if the next element doesn't exist or is invalid
                    if (
                        j == len(scenario.timestamps_seconds) - 1
                        or not rows[j + 1]["valid"]
                    ):
                        continue

                    # Interpolate forwards using next timestep
                    interp_row = {"sim_time": time_expected}

                    next_row = rows[j + 1]
                    next_time = next_row["sim_time"]

                    t = (time_expected - time_current) / (next_time - time_current)
                    interp_row["speed"] = lerp(row["speed"], next_row["speed"], t)
                    interp_row["position_x"] = lerp(
                        row["position_x"], next_row["position_x"], t
                    )
                    interp_row["position_y"] = lerp(
                        row["position_y"], next_row["position_y"], t
                    )

                    h1 = row["heading_rad"]
                    h2 = next_row["heading_rad"]

                    if h2 - h1 > math.pi:
                        h1 += 2 * math.pi
                    elif h1 - h2 > math.pi:
                        h2 += 2 * math.pi

                    interp_row["heading_rad"] = lerp(h1, h2, t) % (2 * math.pi)
                    interp_rows[j] = interp_row

            # Third pass -- filter invalid states, replace interpolated values, convert to ms, constrain angles
            for j in range(num_steps):
                if not rows[j]["valid"]:
                    continue
                if interp_rows[j] is not None:
                    rows[j]["sim_time"] = interp_rows[j]["sim_time"]
                    rows[j]["position_x"] = interp_rows[j]["position_x"]
                    rows[j]["position_y"] = interp_rows[j]["position_y"]
                    rows[j]["heading_rad"] = interp_rows[j]["heading_rad"]
                    rows[j]["speed"] = interp_rows[j]["speed"]
                rows[j]["sim_time"] *= 1000.0
                rows[j]["heading_rad"] = constrain_angle(rows[j]["heading_rad"])
                yield rows[j]

    def _encode_tl_state(self, waymo_state) -> SignalLightState:
        if waymo_state == TrafficSignalLaneState.LANE_STATE_STOP:
            return SignalLightState.STOP
        if waymo_state == TrafficSignalLaneState.LANE_STATE_CAUTION:
            return SignalLightState.CAUTION
        if waymo_state == TrafficSignalLaneState.LANE_STATE_GO:
            return SignalLightState.GO
        if waymo_state == TrafficSignalLaneState.LANE_STATE_ARROW_STOP:
            return SignalLightState.STOP | SignalLightState.ARROW
        if waymo_state == TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION:
            return SignalLightState.CAUTION | SignalLightState.ARROW
        if waymo_state == TrafficSignalLaneState.LANE_STATE_ARROW_GO:
            return SignalLightState.GO | SignalLightState.ARROW
        if waymo_state == TrafficSignalLaneState.LANE_STATE_FLASHING_STOP:
            return SignalLightState.STOP | SignalLightState.FLASHING
        if waymo_state == TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION:
            return SignalLightState.CAUTION | SignalLightState.FLASHING
        return SignalLightState.UNKNOWN

    @property
    def traffic_light_rows(self) -> Generator[Dict, None, None]:
        scenario = self._get_scenario()
        num_steps = len(scenario.timestamps_seconds)
        for i in range(num_steps):
            dynamic_states = scenario.dynamic_map_states[i]
            sim_time = scenario.timestamps_seconds[i] * 1000
            for lane_state in dynamic_states.lane_states:
                row = {
                    "sim_time": sim_time,
                    "state": self._encode_tl_state(lane_state.state).value,
                    "stop_point_x": lane_state.stop_point.x,
                    "stop_point_y": lane_state.stop_point.y,
                    "lane": lane_state.lane,
                }
                yield row

    @staticmethod
    def _lookup_agent_type(agent_type: int) -> int:
        if agent_type == 1:
            return 2  # car
        elif agent_type == 2:
            return 4  # pedestrian
        elif agent_type == 3:
            return 4  # cyclist
        else:
            return 0  # other

    def column_val_in_row(self, row, col_name: str) -> Any:
        return row[col_name]


def import_dataset(
    dataset_spec: types.TrafficHistoryDataset,
    output_path: str,
    map_bbox: Optional[BoundingBox] = None,
):
    """called to pre-process (import) a TrafficHistoryDataset for use by SMARTS"""
    if not dataset_spec.input_path:
        print(f"skipping placeholder dataset spec '{dataset_spec.name}'.")
        return
    output = os.path.join(output_path, f"{dataset_spec.name}.shf")
    if os.path.exists(output):
        os.remove(output)
    source = dataset_spec.source_type
    dataset_dict = dataset_spec.__dict__
    if map_bbox:
        assert dataset_spec.filter_off_map
        dataset_dict["_map_bbox"] = map_bbox
    if source == "NGSIM":
        dataset = NGSIM(dataset_dict, output)
    elif source == "INTERACTION":
        dataset = Interaction(dataset_dict, output)
    elif source == "Waymo":
        dataset = Waymo(dataset_dict, output)
    else:
        raise ValueError(
            f"unsupported TrafficHistoryDataset type: {dataset_spec.source_type}"
        )
    dataset.create_output()


def _check_args(args) -> bool:
    if not args.force and os.path.exists(args.output):
        print("output file already exists\n")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_offset", help="X offset of map", type=float)
    parser.add_argument("--y_offset", help="Y offset of map", type=float)
    parser.add_argument(
        "--force",
        "-f",
        help="Force overwriting output file if it already exists",
        action="store_true",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="""Path to YAML file describing trajectories dataset. YAML file should correspond with types.TrafficHistoryDataset fields.""",
    )
    parser.add_argument(
        "output", type=str, help="SMARTS traffic history file to create"
    )
    args = parser.parse_args()

    if not _check_args(args):
        parser.print_usage()
        sys.exit(-1)

    if args.force and os.path.exists(args.output):
        os.remove(args.output)

    import yaml

    with open(args.dataset, "r") as yf:
        dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

    if not dataset_spec.get("input_path"):
        print(f"skipping placeholder dataset spec at {args.dataset}.")
        sys.exit(0)

    if dataset_spec.get("filter_off_map", False) or dataset_spec.get("flip_y", False):
        print(
            f"cannot use 'filter_off_map' or 'flip_y' as specified in {args.dataset} in command-line usage"
        )
        sys.exit(-1)

    if args.x_offset:
        dataset_spec["x_offset"] = args.x_offset

    if args.y_offset:
        dataset_spec["y_offset"] = args.y_offset

    source = dataset_spec.get("source_type", "NGSIM")
    if source == "NGSIM":
        dataset = NGSIM(dataset_spec, args.output)
    elif source == "Waymo":
        dataset = Waymo(dataset_spec, args.output)
    else:
        dataset = Interaction(dataset_spec, args.output)

    dataset.create_output()
