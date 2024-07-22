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
# This file show how to extract scenario based on metric result
# To use extraction function, download the models_metrics_results.py
from pathlib import Path

from CRITERIA.utils.common import (
    get_argparse_parser,
    get_configuration,
    load_source,
    get_model_metrics_results,
)


def scene_ids_extraction(model_result: dict, scene_ids, metrics_dict: dict):
    require_list = []

    for metric_name, range in metrics_dict.items():
        satisfied_ids = set()
        for scene_id in scene_ids:
            if metric_name == "TAD":
                metric_result = model_result["TAD"][scene_id]["seTAD"]
            elif metric_name == "TAR":
                metric_result = model_result["TAR"][scene_id]["normal_TAR"]
            else:
                metric_result = model_result[metric_name][scene_id]

            if range[0] <= metric_result <= range[1]:
                satisfied_ids.add(scene_id)

        require_list.append(satisfied_ids)

    final_set = set.intersection(*require_list)
    return list(final_set)


if __name__ == "__main__":
    parser = get_argparse_parser(Path(__file__).name)
    args = parser.parse_args()
    config = get_configuration(args)
    program_config: dict = config[Path(__file__).stem]

    models_metrics_results = get_model_metrics_results(config, program_config)

    scene_ids = []
    our_result = load_source(
        config["prediction_models"][program_config["prediction_model"]]
    )
    scene_ids = list(our_result["preds"].keys())

    print(
        scene_ids_extraction(
            models_metrics_results[program_config["prediction_model"]],
            scene_ids,
            metrics_dict=program_config["metric_dict"],
        )
    )
