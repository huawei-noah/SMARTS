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
from pathlib import Path

from ..utils.common import get_argparse_parser, get_configuration, get_file_formatter


def gen_model_metrics_results(config: dict):
    model_result_dict = {}
    program_config: dict = config[Path(__file__).stem]
    metrics_results_dir = (
        program_config.get("metrics_results_dir") or config["metrics_results_dir"]
    )
    metrics_results_dir = metrics_results_dir
    metric_names = program_config.get("metric_names", [])
    file_formatter = get_file_formatter(config["file_formatter"])

    models = program_config.get("model_results_sources")
    for model_name, model_result_sources in models.items():
        result = {}
        for metric_name in metric_names:
            source = model_result_sources.get(f"source_{metric_name}")
            if not source:
                source = Path(
                    program_config["default_format_string"].format(
                        metrics_results_dir=metrics_results_dir,
                        model_name=model_name,
                        metric_name=metric_name,
                        ext=file_formatter.ext,
                    )
                )
            with open(source, "r" + file_formatter.mode) as fp:
                result[metric_name] = file_formatter.formatter.load(fp)
        model_result_dict[model_name] = result

    metrics_results_file = (
        program_config.get("metrics_results_file") or config["metrics_results_file"]
    )
    with open(metrics_results_file, "w" + file_formatter.mode) as handle:
        file_formatter.dump(model_result_dict, handle)
        print(f'"{metrics_results_file}"')


if __name__ == "__main__":
    parser = get_argparse_parser(
        Path(__file__).name, description="Shows the results of a scenario."
    )
    args = parser.parse_args()
    config = get_configuration(args)

    gen_model_metrics_results(config)
