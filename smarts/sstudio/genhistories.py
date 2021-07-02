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

import ijson
import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import as_strided as stride

METERS_PER_FOOT = 0.3048
DEFAULT_LANE_WIDTH = 3.7  # a typical US highway lane is 12ft ~= 3.7m wide


class _TrajectoryDataset:
    def __init__(self, dataset_spec, output):
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
    def scale(self):
        return self._scale

    @property
    def rows(self):
        raise NotImplementedError

    def column_val_in_row(self, row, col_name):
        raise NotImplementedError

    def check_dataset_spec(self, dataset_spec):
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

    def _write_dict(self, curdict, insert_sql, cursor, curkey=""):
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
                   width REAL
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

    def create_output(self, time_precision=3):
        """ time_precision is limit for digits after decimal for sim_time (3 is milisecond precision) """
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
        insert_vehicle_sql = "INSERT INTO Vehicle VALUES (?, ?, ?, ?)"
        insert_traj_sql = "INSERT INTO Trajectory VALUES (?, ?, ?, ?, ?, ?, ?)"
        vehicle_ids = set()
        itcur = dbconxn.cursor()
        for row in self.rows:
            vid = int(self.column_val_in_row(row, "vehicle_id"))
            if vid not in vehicle_ids:
                ivcur = dbconxn.cursor()
                veh_args = (
                    vid,
                    int(self.column_val_in_row(row, "type")),
                    float(self.column_val_in_row(row, "length")) * self.scale,
                    float(self.column_val_in_row(row, "width")) * self.scale,
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
                float(self.column_val_in_row(row, "position_x")) * self.scale,
                float(self.column_val_in_row(row, "position_y")) * self.scale,
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
    def __init__(self, dataset_spec, output):
        super().__init__(dataset_spec, output)
        assert not self._flip_y
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

    @property
    def rows(self):
        with open(self._path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            last_row = None
            for self._next_row in reader:
                if last_row:
                    yield last_row
                last_row = self._next_row
            self._next_row = None
            yield last_row

    @staticmethod
    def _lookup_agent_type(agent_type):
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

    def column_val_in_row(self, row, col_name):
        row_name = self._col_map.get(col_name)
        if row_name:
            return row[row_name]
        if col_name == "length":
            return row.get("length", 0.0)
        if col_name == "width":
            return row.get("width", 0.0)
        if col_name == "type":
            return Interaction._lookup_agent_type(row["agent_type"])
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
                if dm > 0.0:
                    r = math.atan2(dy / dm, dx / dm)
                    return (r - math.pi / 2) % (2 * math.pi)
            # Note: pedestrian track files won't have this
            return float(row.get("psi_rad", 0.0)) - math.pi / 2
        # XXX: should probably check for and handle x_offset_px here too like in NGSIM
        return None


class NGSIM(_TrajectoryDataset):
    def __init__(self, dataset_spec, output):
        super().__init__(dataset_spec, output)
        self._prev_heading = -math.pi / 2

    def _cal_heading(self, window):
        c = window[1, :2]
        n = window[2, :2]
        if any(np.isnan(c)) or any(np.isnan(n)):
            return self._prev_heading
        s = np.linalg.norm(n - c)
        if s == 0.0:
            return self._prev_heading
        vhat = (n - c) / s
        r = math.atan2(vhat[1], vhat[0])
        self._prev_heading = (r - math.pi / 2) % (2 * math.pi)
        return self._prev_heading

    def _cal_speed(self, window):
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
        x_offset = self._dataset_spec.get("x_offset_px", 0) / self.scale
        y_offset = self._dataset_spec.get("y_offset_px", 0) / self.scale

        df["length"] *= METERS_PER_FOOT
        df["width"] *= METERS_PER_FOOT
        df["speed"] *= METERS_PER_FOOT
        df["acceleration"] *= METERS_PER_FOOT
        df["spacing"] *= METERS_PER_FOOT
        df["position_y"] *= METERS_PER_FOOT
        # SMARTS uses center not front
        df["position_x"] = (
            df["position_x"] * METERS_PER_FOOT - 0.5 * df["length"] - x_offset
        )
        if y_offset:
            df["position_x"] = df["position_y"] - y_offset

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
            v = df.loc[same_car, ["position_x", "position_y"]].shift(1).values
            d0, d1 = v.shape
            s0, s1 = v.strides
            headings = [
                self._cal_heading(values)
                for values in stride(v, (d0 - 2, 3, d1), (s0, s0, s1))
            ]
            df.loc[same_car, "heading_rad"] = headings + [headings[-1], headings[-1]]
            # ... and new speeds (based on these smoothed positions)
            # (This also overcomes problem that NGSIM speeds are "instantaneous"
            # and so don't match with dPos/dt, which can affect some models.)
            speeds = [
                self._cal_speed(values)
                for values in stride(v, (d0 - 2, 3, d1), (s0, s0, s1))
            ]
            df.loc[same_car, "speed_discrete"] = speeds + [None, None]

        map_width = self._dataset_spec["map_net"].get("width")
        if map_width:
            valid_x = (df["position_x"] * self.scale).between(0, map_width)
            df = df[valid_x]

        return df

    @property
    def rows(self):
        for t in self._transform_all_data().itertuples():
            yield t

    def column_val_in_row(self, row, col_name):
        if col_name == "speed":
            return row.speed_discrete if row.speed_discrete else row.speed
        return getattr(row, col_name, None)


class OldJSON(_TrajectoryDataset):
    """This exists because SMARTS used to use JSON files for traffic histories.
    We provide this to help people convert these previously-created .json
    history files to the new .shf format."""

    @property
    def rows(self):
        with open(self._dataset_spec["input_path"], "rb") as inf:
            for t, states in ijson.kvitems(inf, "", use_float=True):
                for state in states.values():
                    yield (t, state)

    @staticmethod
    def _lookup_agent_type(agent_type):
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

    def column_val_in_row(self, row, col_name):
        assert len(row) == 2
        if col_name == "sim_time":
            return float(row[0]) * 1000
        state = row[1]
        if col_name in state:
            return state[col_name]
        if col_name == "id":
            return state["vehicle_id"]
        if col_name == "type":
            return OldJSON._lookup_agent_type(state["vehicle_type"])
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


def _check_args(args):
    if not args.force and os.path.exists(args.output):
        print("output file already exists\n")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    if args.old:
        dataset_spec = {"source": "OldJSON", "input_path": args.dataset}
    else:
        with open(args.dataset, "r") as yf:
            dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

    source = dataset_spec.get("source", "NGSIM")
    if source == "NGSIM":
        dataset = NGSIM(dataset_spec, args.output)
    elif source == "OldJSON":
        dataset = OldJSON(dataset_spec, args.output)
    else:
        dataset = Interaction(dataset_spec, args.output)

    dataset.create_output()
