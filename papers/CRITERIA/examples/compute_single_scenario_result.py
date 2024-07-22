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

from CRITERIA.utils.common import get_argparse_parser, get_configuration, load_source
from CRITERIA.utils.scene_metric import scene_metrics_calculator


def main(scene_id, config: dict, show_scene_ids):
    program_config = config[Path(__file__).stem]
    # If want to compute on whole dataset
    our_result = load_source(
        config["prediction_models"][program_config["prediction_model"]]
    )
    if show_scene_ids:
        print(set(our_result["pred"].keys()))
        return

    if scene_id is None:
        try:
            scene_id = next(iter(our_result["preds"].keys()))
        except StopIteration:
            raise IOError("The prediction file is empty.")

    # If want to compute on single scenario
    scene = {"preds": {}, "gts": {}, "cities": {}}

    scene["preds"][scene_id] = our_result["preds"][scene_id]
    scene["gts"][scene_id] = our_result["gts"][scene_id]
    scene["cities"][scene_id] = our_result["cities"][scene_id]

    scene_metric = scene_metrics_calculator(scene, config)
    scene_dict = scene_metric.return_diversity_metrics()
    print(scene_dict)


if __name__ == "__main__":
    parser = get_argparse_parser(
        Path(__file__).name, description="Shows the results of a scenario."
    )
    parser.add_argument(
        "--show_scene_ids", help="Show the scene ids", action="store_true"
    )
    parser.add_argument(
        "--scene_id",
        help="The exact scene id to use.",
        default=None,
    )
    args = parser.parse_args()
    config = get_configuration(args)
    main(args.scene_id, config, args.show_scene_ids)
