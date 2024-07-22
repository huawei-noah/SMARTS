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
import math
import multiprocessing
import multiprocessing.pool
import time
from itertools import zip_longest
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from ..utils.common import get_argparse_parser, get_configuration, load_sources
from ..utils.scene_metric import scene_metrics_calculator


def scenario_classification_avgminFDE(
    model_dict,
    scene_ids,
    config,
    classifications: dict = {
        "easy": (None, 0.1),
        "medium": (0.1, 0.65),
        "hard": (0.65, None),
    },
):  # group scenarios according to different models average FDE
    minFDE_dict = gen_minFDE_dict(model_dict, scene_ids, config)

    # rank according to minFDE
    output = rank_minFDE(classifications, minFDE_dict)

    return output


def rank_minFDE(classifications, minFDE_dict):
    sorted_list = sorted(minFDE_dict.items(), key=lambda x: x[1], reverse=True)

    output = {}
    for classification, (start, end) in classifications.items():
        start_index = None if start is None else int(start * len(sorted_list))
        end_index = None if end is None else int(end * len(sorted_list))
        output[classification] = [
            scene_id for scene_id, _ in sorted_list[start_index:end_index]
        ]

    return output


def gen_minFDE_dict(model_dict, scene_ids, config):
    minFDE_dict = {}
    # extract minFDE
    for id in tqdm(scene_ids):
        minFDE_list = []
        for model_name, result in model_dict.items():
            scene_result = {}
            scene_result.clear()
            scene_result["preds"] = {}
            scene_result["gts"] = {}
            scene_result["cities"] = {}
            scene_result["preds"][id] = result["preds"][id]
            scene_result["gts"][id] = result["gts"][id]
            scene_result["cities"][id] = result["cities"][id]

            scene_metric = scene_metrics_calculator(scene_result, config)
            minFDE_list.append(scene_metric.get_minFDE())

        minFDE_dict[id] = sum(minFDE_list) / len(minFDE_list)
    return minFDE_dict


def _wait_async(results_async, count, check_rate):
    ready, active = [], []
    while len(ready) == 0:
        for r in results_async:
            (active, ready)[r.ready()].append(r)
        if len(results_async) < count:
            break
        time.sleep(check_rate)
    return ready, active


def distributed_scenario_classification_avgminFDE(
    model_dict,
    scene_ids,
    config,
    classifications: dict = {
        "easy": (None, 0.1),
        "medium": (0.1, 0.65),
        "hard": (0.65, None),
    },
    count: int = 6,
    chunk: int = 400,
    check_rate: float = 0.2,
):
    v = math.ceil(len(scene_ids) / 6)
    results_async: List[AsyncResult] = []
    current_minFDE_dict = {}
    ready, active = [], []
    with Pool(count) as pool:
        for c in range(0, len(scene_ids), chunk):
            ar = pool.apply_async(
                gen_minFDE_dict,
                dict(
                    model_dict=model_dict,
                    scene_ids=scene_ids[c : c + chunk],
                    config=config,
                ),
            )
            results_async.append(ar)
            ready, active = _wait_async(results_async, count, check_rate)
            results_async = active
            for ar in ready:
                current_minFDE_dict.update(ar.get())

        while len(active):
            ready, active = _wait_async(results_async, count, check_rate)
            for ar in ready:
                current_minFDE_dict.update(ar.get())

    # rank according to minFDE
    output = rank_minFDE(classifications, current_minFDE_dict)

    return output


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).stem)
    args = parser.parse_args()
    config = get_configuration(args)
    program_config = config[Path(__file__).stem]

    metrics_section = config["metrics"]
    trajectory_count = metrics_section["trajectory_count"]
    model_dict = load_sources(config["prediction_models"])
    prediction_result = model_dict[program_config["prediction_model"]]
    scene_ids = list(set(prediction_result["preds"].keys()))

    classifications = scenario_classification_avgminFDE(
        model_dict=model_dict,
        scene_ids=scene_ids,
        config=config,
        classifications=program_config["minFDE_classifications"],
        # count=program_config["dist"]["count"],
        # chunk=program_config["dist"]["chunks"],
        # check_rate=program_config["dist"]["check_rate"],
    )

    avgminFDE_excel = program_config["avgminFDE_excel"]
    Path(avgminFDE_excel).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(avgminFDE_excel, "openpyxl", mode="w") as ew:
        df = pd.DataFrame(
            list(zip_longest(*classifications.values())),
            columns=list(classifications.keys()),
        )
        df.to_excel(ew, sheet_name="list_1", index=False)
        print(avgminFDE_excel)
