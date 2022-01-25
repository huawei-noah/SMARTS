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

import argparse
import csv
import logging
import math
import os
import sqlite3
import struct
import sys
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union

import ijson
import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import as_strided as stride
from numpy.lib.stride_tricks import sliding_window_view

from smarts.core.utils.math import vec_to_radians

try:
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    print(sys.exc_info())
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
        self._path = dataset_spec["input_path"]
        real_lane_width_m = dataset_spec.get("real_lane_width_m", DEFAULT_LANE_WIDTH)
        lane_width = dataset_spec.get("map_net", {}).get(
            "lane_width", real_lane_width_m
        )
        self._scale = lane_width / real_lane_width_m
        self._flip_y = dataset_spec.get("flip_y", False)
        self._swap_xy = dataset_spec.get("swap_xy", False)

    @property
    def scale(self) -> float:
        """The base scale based on the ratio of map lane size to real lane size."""
        return self._scale

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
            errmsg = "'input_path' field is required in dataset yaml."
        elif dataset_spec.get("flip_y"):
            if dataset_spec.get("source") != "NGSIM":
                errmsg = "'flip_y' option only supported for NGSIM datasets."
            elif not dataset_spec.get("map_net", {}).get("max_y"):
                errmsg = "'map_net:max_y' is required if 'flip_y' option used."
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
                   FOREIGN KEY (vehicle_id) REFERENCES Vehicles(id)
               ) WITHOUT ROWID"""
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
            # Ignore datapoints with NaNs because the rolling window code used by
            # NGSIM can leave about a kernel-window's-worth of NaNs at the end.
            if not any(a is not None and np.isnan(a) for a in traj_args):
                itcur.execute(insert_traj_sql, traj_args)
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
        self._heading_min_speed = dataset_spec.get("heading_inference_min_speed", 0.22)
        self._prev_heading = None
        self._next_row = None
        # See: https://interaction-dataset.com/details-and-format
        # position and length/width are in meters.
        # Note: track_id will be like "P12" for pedestrian tracks.  (TODO)
        self._col_map = {
            "vehicle_id": "track_id",
            "sim_time": "timestamp_ms",
            "position_x": "y" if self._swap_xy else "x",
            "position_y": "x" if self._swap_xy else "y",
        }

    def check_dataset_spec(self, dataset_spec: Dict[str, Any]):
        super().check_dataset_spec(dataset_spec)
        hiw = dataset_spec.get("heading_inference_window", 2)
        if hiw != 2:
            # Adding support for this would require changing the rows() generator
            # (since we're not using Pandas here like we are for NGSIM).
            # So wait until if/when users request it...
            raise ValueError(
                "heading_inference_window not yet supported for Interaction datasets."
            )

    @property
    def rows(self) -> Generator[Dict, None, None]:
        with open(self._path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            last_row = None
            for self._next_row in reader:
                if last_row:
                    yield last_row
                last_row = self._next_row
            self._next_row = None
            if last_row:
                yield last_row

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

    def column_val_in_row(self, row, col_name: str) -> Any:
        row_name = self._col_map.get(col_name)
        if row_name:
            return row[row_name]
        if col_name == "length":
            return row.get("length", 0.0)
        if col_name == "width":
            return row.get("width", 0.0)
        if col_name == "type":
            return self._lookup_agent_type(row["agent_type"])
        if col_name == "speed":
            if self._next_row:
                # XXX: could try to divide by sim_time delta here instead of assuming .1s
                dx = (float(self._next_row["x"]) - float(row["x"])) / 0.1
                dy = (float(self._next_row["y"]) - float(row["y"])) / 0.1
            else:
                dx, dy = float(row["vx"]), float(row["vy"])
            return np.linalg.norm((dx, dy))
        if col_name == "heading_rad":
            if self._next_row:
                dx = float(self._next_row["x"]) - float(row["x"])
                dy = float(self._next_row["y"]) - float(row["y"])
                dm = np.linalg.norm((dx, dy))
                if dm != 0.0 and dm > self._heading_min_speed:
                    new_heading = vec_to_radians((dx, dy))
                    if self._max_angular_velocity and self._prev_heading is not None:
                        # XXX: could try to divide by sim_time delta here instead of assuming .1s
                        angular_velocity = (new_heading - self._prev_heading) / 0.1
                        if abs(angular_velocity) > self._max_angular_velocity:
                            new_heading = (
                                self._prev_heading
                                + np.sign(angular_velocity)
                                * self._max_angular_velocity
                                * 0.1
                            )
                    self._prev_heading = new_heading
                    return new_heading
            # Note: pedestrian track files won't have this
            self._prev_heading = float(row.get("psi_rad", 0.0)) - math.pi / 2
            return self._prev_heading
        # XXX: should probably check for and handle x_offset_px here too like in NGSIM
        return None


class NGSIM(_TrajectoryDataset):
    """A tool for conversion of a NGSIM dataset for use within SMARTS."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        super().__init__(dataset_spec, output)
        self._prev_heading = 3 * math.pi / 2
        self._max_angular_velocity = dataset_spec.get("max_angular_velocity", None)
        self._heading_window = dataset_spec.get("heading_inference_window", 2)
        # .22 corresponds to roughly 5mph.
        self._heading_min_speed = dataset_spec.get("heading_inference_min_speed", 0.22)

    def check_dataset_spec(self, dataset_spec: Dict[str, Any]):
        super().check_dataset_spec(dataset_spec)
        hiw = dataset_spec.get("heading_inference_window", 2)
        # 11 is a semi-arbitrary max just to keep things "sane".
        if not 2 <= hiw <= 11:
            raise ValueError("heading_inference_window must be between 2 and 11")

    def _cal_heading(self, window_param) -> float:
        window = window_param[0]
        new_heading = 0
        prev_heading = None
        den = 0
        for w in range(self._heading_window - 1):
            c = window[w, :2]
            n = window[w + 1, :2]
            if any(np.isnan(c)) or any(np.isnan(n)):
                if prev_heading is not None:
                    new_heading += prev_heading
                    den += 1
                continue
            s = np.linalg.norm(n - c)
            ispeed = window[w, 2]
            if s == 0.0 or (
                self._heading_min_speed is not None
                and (s < self._heading_min_speed or ispeed < self._heading_min_speed)
            ):
                if prev_heading is not None:
                    new_heading += prev_heading
                    den += 1
                continue
            vhat = (n - c) / s
            inst_heading = vec_to_radians(vhat)
            if prev_heading is not None:
                if inst_heading == 0 and prev_heading > math.pi:
                    inst_heading = 2 * math.pi
                if self._max_angular_velocity:
                    # XXX: could try to divide by sim_time delta here instead of assuming .1s
                    angular_velocity = (inst_heading - prev_heading) / 0.1
                    if abs(angular_velocity) > self._max_angular_velocity:
                        inst_heading = (
                            prev_heading
                            + np.sign(angular_velocity)
                            * self._max_angular_velocity
                            * 0.1
                        )
            den += 1
            new_heading += inst_heading
            prev_heading = inst_heading
        if den > 0:
            new_heading /= den
        else:
            new_heading = self._prev_heading
        self._prev_heading = new_heading
        return new_heading % (2 * math.pi)

    def _cal_speed(self, window) -> Optional[float]:
        c = window[1, :2]
        n = window[2, :2]
        badc = any(np.isnan(c))
        badn = any(np.isnan(n))
        if badc or badn:
            return None
        # XXX: could try to divide by sim_time delta here instead of assuming .1s
        return np.linalg.norm(n - c) / 0.1

    def _transform_all_data(self):
        self._log.debug("transforming NGSIM data")
        df = pd.read_csv(
            self._path,
            sep=r"\s+",
            header=None,
            names=(
                "vehicle_id",
                "frame_id",  # 1 frame per .1s
                "total_frames",
                "global_time",  # msecs
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
            ),
        )

        df["sim_time"] = df["global_time"] - min(df["global_time"])

        # offset of the map from the data...
        x_margin = self._dataset_spec.get("x_margin_px", 0) / self.scale
        y_margin = self._dataset_spec.get("y_margin_px", 0) / self.scale

        df["length"] *= METERS_PER_FOOT
        df["width"] *= METERS_PER_FOOT
        df["speed"] *= METERS_PER_FOOT
        df["acceleration"] *= METERS_PER_FOOT
        df["spacing"] *= METERS_PER_FOOT
        df["position_y"] *= METERS_PER_FOOT
        # SMARTS uses center not front
        df["position_x"] = (
            df["position_x"] * METERS_PER_FOOT - 0.5 * df["length"] - x_margin
        )
        if y_margin:
            df["position_x"] = df["position_y"] - y_margin

        if self._flip_y:
            max_y = self._dataset_spec["map_net"]["max_y"]
            df["position_y"] = (max_y / self.scale) - df["position_y"]

        # Use moving average to smooth positions...
        df.sort_values("sim_time", inplace=True)  # just in case it wasn't already...
        k = 15  # kernel size for positions
        for vehicle_id in set(df["vehicle_id"]):
            same_car = df["vehicle_id"] == vehicle_id
            df.loc[same_car, "position_x"] = (
                df.loc[same_car, "position_x"]
                .rolling(window=k)
                .mean()
                .shift(1 - k)
                .values
            )
            df.loc[same_car, "position_y"] = (
                df.loc[same_car, "position_y"]
                .rolling(window=k)
                .mean()
                .shift(1 - k)
                .values
            )
            # and compute heading with (smaller) rolling window (=3) too..
            shift = int(self._heading_window / 2)
            pad = self._heading_window - shift - 1
            v = df.loc[same_car, ["position_x", "position_y", "speed"]].values
            v = np.insert(v, 0, [[np.nan, np.nan, np.nan]] * shift, axis=0)
            headings = [
                self._cal_heading(values)
                for values in sliding_window_view(v, (self._heading_window, 3))
            ]
            df.loc[same_car, "heading_rad"] = headings + [headings[-1]] * pad
            # ... and new speeds (based on these smoothed positions)
            # (This also overcomes problem that NGSIM speeds are "instantaneous"
            # and so don't match with dPos/dt, which can affect some models.)
            v = df.loc[same_car, ["position_x", "position_y"]].shift(1).values
            d0, d1 = v.shape
            s0, s1 = v.strides
            speeds = [
                self._cal_speed(values)
                for values in stride(v, (d0 - 2, 3, d1), (s0, s0, s1))
            ]
            df.loc[same_car, "speed_discrete"] = speeds + [None, None]

        map_width = self._dataset_spec["map_net"].get("width")
        if map_width:
            valid_x = (df["position_x"] * self.scale).between(
                df["length"] / 2, map_width - df["length"] / 2
            )
            df = df[valid_x]

        return df

    @property
    def rows(self) -> Generator[Dict, None, None]:
        for t in self._transform_all_data().itertuples():
            yield t

    def column_val_in_row(self, row, col_name: str) -> Any:
        if col_name == "speed":
            return row.speed_discrete if row.speed_discrete else row.speed
        return getattr(row, col_name, None)


class OldJSON(_TrajectoryDataset):
    """This exists because SMARTS used to use JSON files for traffic histories.
    We provide this to help people convert these previously-created .json
    history files to the new .shf format."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        from warnings import warn

        warn(
            f"The {self.__class__.__name__} class has been deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(dataset_spec, output)

    @property
    def rows(self) -> Generator[Tuple, None, None]:
        with open(self._dataset_spec["input_path"], "rb") as inf:
            for t, states in ijson.kvitems(inf, "", use_float=True):
                for state in states.values():
                    yield (t, state)

    def _lookup_agent_type(self, agent_type: Union[int, str]) -> int:
        if isinstance(agent_type, int):
            return agent_type
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

    def column_val_in_row(self, row: Tuple, col_name: str) -> Any:
        assert len(row) == 2
        if col_name == "sim_time":
            return float(row[0]) * 1000
        state = row[1]
        if col_name in state:
            return state[col_name]
        if col_name == "id":
            return state["vehicle_id"]
        if col_name == "type":
            return self._lookup_agent_type(state["vehicle_type"])
        if col_name == "length":
            return state.get("vehicle_length", 0.0)
        if col_name == "width":
            return state.get("vehicle_width", 0.0)
        if col_name.startswith("position_x"):
            return state["position"][0]
        if col_name.startswith("position_y"):
            return state["position"][1]
        if col_name == "heading_rad":
            return state.get("heading", -math.pi / 2)
        return None


class Waymo(_TrajectoryDataset):
    """A tool for conversion of a Waymo dataset for use within SMARTS."""

    def __init__(self, dataset_spec: Dict[str, Any], output: str):
        super().__init__(dataset_spec, output)

    @staticmethod
    def read_dataset(path: str) -> Generator[bytes, None, None]:
        """Iterate over the records in a TFRecord file and return the bytes of each record.

        path: The path to the TFRecord file
        """
        with open(path, "rb") as f:
            while True:
                length_bytes = f.read(8)
                if len(length_bytes) != 8:
                    return
                record_len = int(struct.unpack("Q", length_bytes)[0])
                _ = f.read(4)  # masked_crc32_of_length (ignore)
                record_data = f.read(record_len)
                _ = f.read(4)  # masked_crc32_of_data (ignore)
                yield record_data

    @property
    def rows(self) -> Generator[Dict, None, None]:
        def lerp(a: float, b: float, t: float) -> float:
            return t * (b - a) + a

        def constrain_angle(angle: float) -> float:
            """Constrain to [-pi, pi]"""
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle -= 2 * math.pi
            return angle

        if "scenario_id" not in self._dataset_spec:
            errmsg = "Dataset spec requires scenario_id to be set"
            self._log.error(errmsg)
            raise ValueError(errmsg)
        scenario_id = self._dataset_spec["scenario_id"]

        # Loop over the scenarios in the TFRecord and check its ID for a match
        scenario = None
        dataset = Waymo.read_dataset(self._dataset_spec["input_path"])
        for record in dataset:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                scenario = parsed_scenario
                break

        if not scenario:
            errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
            self._log.error(errmsg)
            raise ValueError(errmsg)

        for i in range(len(scenario.tracks)):
            vehicle_id = scenario.tracks[i].id
            vehicle_type = self._lookup_agent_type(scenario.tracks[i].object_type)
            num_steps = len(scenario.timestamps_seconds)
            rows = []

            # First pass -- extract data
            for j in range(num_steps):
                obj_state = scenario.tracks[i].states[j]
                vel = np.array([obj_state.velocity_x, obj_state.velocity_y])

                row = {}
                row["valid"] = obj_state.valid
                row["vehicle_id"] = vehicle_id
                row["type"] = vehicle_type
                row["length"] = obj_state.length
                row["height"] = obj_state.height
                row["width"] = obj_state.width
                row["sim_time"] = scenario.timestamps_seconds[j]
                row["position_x"] = obj_state.center_x
                row["position_y"] = obj_state.center_y
                row["heading_rad"] = obj_state.heading - math.pi / 2
                row["speed"] = np.linalg.norm(vel)
                row["lane_id"] = 0
                row["is_ego_vehicle"] = 1 if i == scenario.sdc_track_index else 0
                rows.append(row)

            # Second pass -- align timesteps to 10 Hz and interpolate trajectory data if needed
            interp_rows = [None] * num_steps
            for j in range(num_steps):
                row = rows[j]
                timestep = 0.1
                time_current = row["sim_time"]
                time_expected = round(j * timestep, 3)
                time_error = time_current - time_expected

                if not row["valid"] or time_error == 0:
                    continue

                if time_error > 0:
                    # We can't interpolate if the previous element doesn't exist or is invalid
                    if j == 0 or not rows[j - 1]["valid"]:
                        continue

                    # Interpolate backwards using previous timestep
                    interp_row = {}
                    interp_row["sim_time"] = time_expected

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
                    interp_row = {}
                    interp_row["sim_time"] = time_expected

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
                    interp_row["heading_rad"] = lerp(
                        row["heading_rad"], next_row["heading_rad"], t
                    )
                    interp_rows[j] = interp_row

            # Third pass -- filter invalid states, replace interpolated values, convert to ms, constrain angles
            for j in range(num_steps):
                if rows[j]["valid"] == False:
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
        "--old",
        "--json",
        help="Input is an old SMARTS traffic history in JSON format as opposed to a YAML dataset spec.",
        action="store_true",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="""Path to YAML file describing original trajectories dataset. See SMARTS Issue #732 for YAML file options.
                Note: if --old is used, this path is expected to point to an old JSON traffic history to be converted.""",
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

    if args.old:
        dataset_spec = {"source": "OldJSON", "input_path": args.dataset}
    else:
        with open(args.dataset, "r") as yf:
            dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

    if args.x_offset:
        dataset_spec["x_offset"] = args.x_offset

    if args.y_offset:
        dataset_spec["y_offset"] = args.y_offset

    source = dataset_spec.get("source", "NGSIM")
    if source == "NGSIM":
        dataset = NGSIM(dataset_spec, args.output)
    elif source == "Waymo":
        dataset = Waymo(dataset_spec, args.output)
    elif source == "OldJSON":
        dataset = OldJSON(dataset_spec, args.output)
    else:
        dataset = Interaction(dataset_spec, args.output)

    dataset.create_output()
