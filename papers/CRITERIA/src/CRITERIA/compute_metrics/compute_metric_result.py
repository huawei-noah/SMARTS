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
# This file give examples on how to calculate model metric result
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from CRITERIA.utils.common import (
    get_argparse_parser,
    get_configuration,
    get_file_formatter,
    load_sources,
)
from CRITERIA.utils.scene_metric import scene_metrics_calculator


def main(
    config: dict, TNT: bool, LaneGCN: bool, HiVT: bool, FTGN: bool, mmTransformer: bool
):

    models_to_use = dict(
        TNT=TNT,
        LaneGCN=LaneGCN,
        HiVT=HiVT,
        FTGN=FTGN,
        mmTransformer=mmTransformer,
    )

    program_config = config[Path(__file__).stem]
    # If want to compute on whole dataset
    prediction_models: dict = load_sources(config["prediction_models"])
    scene_ids = set(
        prediction_models[next(iter(prediction_models.keys()))]["preds"].keys()
    )

    format_str: str = (
        program_config.get("default_format_string") or config["default_format_string"]
    )
    metrics_results_dir = (
        program_config.get("metrics_results_dir") or config["metrics_results_dir"]
    )
    file_formatter = get_file_formatter(config["file_formatter"])
    for model_name in (mn for mn, use in models_to_use.items() if use):
        metrics_result = {}
        pred_model = prediction_models[model_name]

        # If want to compute on single scenario
        for scene_id in tqdm(scene_ids, desc=model_name):
            scene = {"preds": {}, "gts": {}, "cities": {}}
            scene["preds"][scene_id] = pred_model["preds"][scene_id]
            scene["gts"][scene_id] = pred_model["gts"][scene_id]
            scene["cities"][scene_id] = pred_model["cities"][scene_id]

            scene_metric = scene_metrics_calculator(scene, config)
            scene_dict = scene_metric.return_paper_metrics()
            for metric_name in scene_dict:
                metrics_result.setdefault(metric_name, {})[scene_id] = scene_dict[
                    metric_name
                ]

        for metric_name in metrics_result:
            output_file = Path(
                format_str.format(
                    metrics_results_dir=metrics_results_dir,
                    model_name=model_name,
                    metric_name=metric_name,
                    ext=file_formatter.ext,
                )
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w" + file_formatter.mode) as handle:
                file_formatter.formatter.dump(metrics_result[metric_name], handle)
            print(f'"{output_file}"')


if __name__ == "__main__":
    parser = get_argparse_parser(
        Path(__file__).name, description="Generates scenario data."
    )
    parser.add_argument("--TNT", help="Use TNT", action="store_true")
    parser.add_argument("--LaneGCN", help="Use LaneGCN", action="store_true")
    parser.add_argument("--HiVT", help="Use HiVT", action="store_true")
    parser.add_argument("--FTGN", help="Use FTGN", action="store_true")
    parser.add_argument(
        "--mmTransformer", help="Use mmTransformer", action="store_true"
    )
    args = parser.parse_args()
    config = get_configuration(args)
    main(config, args.TNT, args.LaneGCN, args.HiVT, args.FTGN, args.mmTransformer)
