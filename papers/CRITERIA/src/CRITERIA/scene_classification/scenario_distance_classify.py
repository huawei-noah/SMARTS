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

import numpy as np
import pandas as pd
from argoverse.data_loading.argoverse_forecasting_loader import (
    ArgoverseForecastingLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm

from ..utils.common import get_argparse_parser, get_configuration, load_sources
from ..utils.scene_metric import scene_metrics_calculator


def scenario_classification_distance(
    distance_barrier, prediction_result, scene_ids
):  # group scenarios according to travel distance:
    scene_ids = prediction_result["preds"].keys()
    dis_dict = {}
    # calculate distance
    for id in tqdm(scene_ids):
        gt = prediction_result["gts"][id]
        total_dis = 0
        for i in range(len(gt) - 1):
            total_dis += np.linalg.norm(gt[i] - gt[i + 1])

        dis_dict[id] = total_dis

    sorted_list = sorted(dis_dict.items(), key=lambda x: x[1], reverse=True)

    long_list = []
    short_list = []
    for scene_id, distance in sorted_list:
        if distance > distance_barrier:
            long_list.append(scene_id)
        else:
            short_list.append(scene_id)
    return long_list, short_list


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).stem)
    args = parser.parse_args()
    config = get_configuration(args)
    program_config = config[Path(__file__).stem]

    metrics_section = config["metrics"]
    model_dict = load_sources(config["prediction_models"])
    prediction_result = model_dict[program_config["prediction_model"]]
    scene_ids = set(prediction_result["preds"].keys())
    long_list, short_list = scenario_classification_distance(
        program_config["distance_barrier"],
        prediction_result=prediction_result,
        scene_ids=scene_ids,
    )

    long_short_excel = program_config["long_short_excel"]
    Path(long_short_excel).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(long_short_excel, "openpyxl", mode="w") as ew:
        df = pd.DataFrame(
            list(zip_longest(long_list, short_list)), columns=["long", "short"]
        )
        df.to_excel(ew, sheet_name="list_1", index=False)
        print(long_short_excel)
