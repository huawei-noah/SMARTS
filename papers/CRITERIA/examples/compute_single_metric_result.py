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
from CRITERIA.utils.metrics import metrics_calculator


def main(config: dict):
    program_config = config[Path(__file__).stem]
    # If want to compute on whole dataset
    our_result = load_source(
        config["prediction_models"][program_config["prediction_model"]]
    )

    metric = metrics_calculator(our_result, config)
    metric_result = (
        metric.return_diversity_metrics()
    )  # return all diversity/admissibility metrics take time
    print(metric_result)


if __name__ == "__main__":
    # Note metrics/new_predictions/TNT/TNT.pickle
    parser = get_argparse_parser(Path(__file__).name)
    args = parser.parse_args()
    config = get_configuration(args)

    main(config)
