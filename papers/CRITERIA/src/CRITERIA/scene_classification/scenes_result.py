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
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.common import get_argparse_parser, get_configuration, load_sources
from ..utils.metrics import metrics_calculator


def main(config: dict):
    program_config: dict = config[Path(__file__).stem]
    models_metrics_results = {}
    with open(config["metrics_results_file"], "rb") as handle:
        models_metrics_results = pkl.load(handle)
    model_dict = load_sources(config["prediction_models"])

    scene_df = pd.read_excel(
        program_config.get("scene_df_file") or config["scene_df_file"],
        sheet_name="list_1",
        engine="openpyxl",
    )

    easy_cruise_long = set()
    easy_cruise_short = set()
    easy_inter_long = set()
    easy_inter_short = set()

    middle_cruise_long = set()
    middle_cruise_short = set()
    middle_inter_long = set()
    middle_inter_short = set()

    hard_cruise_long = set()
    hard_cruise_short = set()
    hard_inter_long = set()
    hard_inter_short = set()

    for scene_id in scene_df["easy_cruise_long"]:
        if scene_id >= 0:
            easy_cruise_long.add(int(scene_id))

    for scene_id in scene_df["easy_cruise_short"]:
        if scene_id >= 0:
            easy_cruise_short.add(int(scene_id))

    for scene_id in scene_df["easy_inter_long"]:
        if scene_id >= 0:
            easy_inter_long.add(int(scene_id))

    for scene_id in scene_df["easy_inter_short"]:
        if scene_id >= 0:
            easy_inter_short.add(int(scene_id))

    for scene_id in scene_df["middle_cruise_long"]:
        if scene_id >= 0:
            middle_cruise_long.add(int(scene_id))

    for scene_id in scene_df["middle_cruise_short"]:
        if scene_id >= 0:
            middle_cruise_short.add(int(scene_id))

    for scene_id in scene_df["middle_inter_long"]:
        if scene_id >= 0:
            middle_inter_long.add(int(scene_id))

    for scene_id in scene_df["middle_inter_short"]:
        if scene_id >= 0:
            middle_inter_short.add(int(scene_id))

    for scene_id in scene_df["hard_cruise_long"]:
        if scene_id >= 0:
            hard_cruise_long.add(int(scene_id))

    for scene_id in scene_df["hard_cruise_short"]:
        if scene_id >= 0:
            hard_cruise_short.add(int(scene_id))

    for scene_id in scene_df["hard_inter_long"]:
        if scene_id >= 0:
            hard_inter_long.add(int(scene_id))

    for scene_id in scene_df["hard_inter_short"]:
        if scene_id >= 0:
            hard_inter_short.add(int(scene_id))

    scene_dict = {
        "easy_cruise_long": easy_cruise_long,
        "easy_cruise_short": easy_cruise_short,
        "easy_inter_long": easy_inter_long,
        "easy_inter_short": easy_inter_short,
        "middle_cruise_long": middle_cruise_long,
        "middle_cruise_short": middle_cruise_short,
        "middle_inter_long": middle_inter_long,
        "middle_inter_short": middle_inter_short,
        "hard_cruise_long": hard_cruise_long,
        "hard_cruise_short": hard_cruise_short,
        "hard_inter_long": hard_inter_long,
        "hard_inter_short": hard_inter_short,
    }

    result_dict = {
        "TNT": {"hard": {}, "middle": {}, "easy": {}},
        "LaneGCN": {"hard": {}, "middle": {}, "easy": {}},
        "HiVT": {"hard": {}, "middle": {}, "easy": {}},
        "FTGN": {"hard": {}, "middle": {}, "easy": {}},
        "mmTransformer": {"hard": {}, "middle": {}, "easy": {}},
    }

    for scene_name, group in scene_dict.items():
        print(scene_name)
        for model_name, metrics_result in models_metrics_results.items():
            print(model_name)
            result = model_dict[model_name]

            minFDE_list = []
            minADE_list = []
            RF_list = []
            DAO_list = []
            DAC_list = []
            minASD_list = []
            minFSD_list = []
            TAD_list = []
            TDD_list = []
            TAR_list = []
            seTAD_list = []
            # RF calculation
            scene = {}
            scene.clear()
            scene["preds"] = {}
            scene["gts"] = {}
            scene["cities"] = {}
            for scene_id in group:
                minFDE_list.append(metrics_result["minFDE"][scene_id])
                minADE_list.append(metrics_result["minADE"][scene_id])
                DAO_list.append(metrics_result["DAO"][scene_id])
                DAC_list.append(metrics_result["DAC"][scene_id])

                minASD_list.append(metrics_result["minASD"][scene_id])
                minFSD_list.append(metrics_result["minFSD"][scene_id])
                TAD_list.append(metrics_result["TAD"][scene_id]["seTAD"])
                TDD_list.append(metrics_result["TDD"][scene_id])
                TAR_list.append(metrics_result["TAR"][scene_id]["normal_TAR"])

                # RF calculation
                scene["preds"][scene_id] = result["preds"][scene_id]
                scene["gts"][scene_id] = result["gts"][scene_id]
                scene["cities"][scene_id] = result["cities"][scene_id]

            # RF calculation
            scene_metric = metrics_calculator(scene, config)
            rf = scene_metric.get_RF_avgFDE()[0]

            current_result_dict = {
                "minFDE": np.average(minFDE_list),
                "minADE": np.average(minADE_list),
                "RF": rf,
                "minFSD": sum(minFSD_list) / len(minFSD_list),
                "minASD": sum(minASD_list) / len(minASD_list),
                "TAD": sum(TAD_list) / len(TAD_list),
                "TDD": sum(TDD_list) / len(TDD_list),
                "DAO": sum(DAO_list) / len(DAO_list),
                "DAC": np.average(DAC_list),
                "TAR": sum(TAR_list) / len(TAR_list),
            }

            print(current_result_dict)


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).stem)
    args = parser.parse_args()

    config = get_configuration(args)

    main(config)
