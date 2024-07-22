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
from itertools import product, zip_longest
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
from .scenario_difficulty_classify import scenario_classification_avgminFDE
from .scenario_distance_classify import scenario_classification_distance
from .scenario_intersect_classify import scenario_classification_intersection

if __name__ == "__main__":
    program = Path(__file__).stem
    parser = get_argparse_parser(program)
    args = parser.parse_args()
    config = get_configuration(args)
    program_config = config[program]

    df_d = pd.read_excel(
        program_config.get("avgminFDE_excel"),
        sheet_name="list_1",
        engine="openpyxl",
    )
    df_ls = pd.read_excel(
        program_config.get("long_short_excel"),
        sheet_name="list_1",
        engine="openpyxl",
    )
    df_ci = pd.read_excel(
        program_config.get("cruise_intersect_excel"),
        sheet_name="list_1",
        engine="openpyxl",
    )

    twelve_scenes_classification = {}
    for dif in reversed(df_d.columns):
        for inter in df_ci:
            # to avoid additional intersections
            c_intersection = np.intersect1d(df_d[dif], df_ci[inter])
            for l in df_ls:
                twelve_scenes_classification[f"{dif}_{inter}_{l}"] = np.intersect1d(
                    c_intersection, df_ls[l]
                )

    twelve_scenes_excel = program_config.get("scene_df_file") or config["scene_df_file"]
    Path(twelve_scenes_excel).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(twelve_scenes_excel, "openpyxl", mode="w") as ew:
        df = pd.DataFrame(
            zip_longest(*twelve_scenes_classification.values()),
            columns=list(twelve_scenes_classification.keys()),
        )
        df.to_excel(ew, sheet_name="list_1", index=False)
        print(twelve_scenes_excel)
