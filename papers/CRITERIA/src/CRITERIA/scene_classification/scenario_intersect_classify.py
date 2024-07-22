# MIT License
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
# classified scnearios by distance, difficulty and type
from itertools import zip_longest
from pathlib import Path

import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm

from ..utils.common import get_argparse_parser, get_configuration, load_sources


def scenario_classification_intersection(trajectory_count, afl):
    am = ArgoverseMap()
    intersect_list = set()
    cruise_list = set()
    for i in tqdm(range(0, len(afl))):
        seq_path = afl.seq_list[i]
        # ground truth trajectory
        agent_traj = afl.get(seq_path).agent_traj[trajectory_count:]
        city = afl[i].city

        lane_list = []
        # retrieve lane segments in 1m radius of agent position
        for coord in agent_traj:
            lane_list.extend(am.get_lane_ids_in_xy_bbox(coord[0], coord[1], city, 1))

        lane_set = set(lane_list)

        intersect = []
        left = []
        right = []
        seq_id = int(Path(seq_path.name).stem)
        for lane_id in lane_set:
            intersect.append(am.lane_is_in_intersection(lane_id, city))
            left.append(am.get_lane_turn_direction(lane_id, city) == "LEFT")
            right.append(am.get_lane_turn_direction(lane_id, city) == "RIGHT")

        # left, right, intersection are in intersection category
        if sum(intersect) != 0:
            intersect_list.add(seq_id)
        if sum(left) != 0:
            intersect_list.add(seq_id)
        if sum(right) != 0:
            intersect_list.add(seq_id)
        if sum(left) == 0 and sum(right) == 0 and sum(intersect) == 0:
            cruise_list.add(seq_id)

    return list(intersect_list), list(cruise_list)


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).stem)
    args = parser.parse_args()
    config = get_configuration(args)
    program_config = config[Path(__file__).stem]

    metrics_section = config["metrics"]
    trajectory_count = metrics_section["trajectory_count"]

    root_dir = config["argoverse_data_dir"]  # change to your argoverse data path
    afl = ArgoverseForecastingLoader(root_dir)
    intersect_list, cruise_list = scenario_classification_intersection(
        trajectory_count=trajectory_count, afl=afl
    )

    cruise_intersect_excel = program_config["cruise_intersect_excel"]
    Path(cruise_intersect_excel).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cruise_intersect_excel, "openpyxl", mode="w") as ew:
        df = pd.DataFrame(
            list(zip_longest(cruise_list, intersect_list)),
            columns=["cruise", "inter"],
        )
        df.to_excel(ew, sheet_name="list_1", index=False)
        print(cruise_intersect_excel)
