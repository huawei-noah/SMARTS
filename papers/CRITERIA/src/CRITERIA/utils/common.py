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
from __future__ import annotations

import argparse
import importlib.resources
import pickle as pkl
from ast import Module
from dataclasses import dataclass
from typing import Union

import numpy as np
import toml

REFERENCE_FRAME = 19
import CRITERIA.resources


@dataclass
class FileFormatter:
    name: str
    formatter: Union[Module]
    ext: str
    mode: str


def get_argparse_parser(program_name: str, *args, **kwargs) -> argparse.ArgumentParser:
    program_name = f"(CRITERIA) {program_name}"
    parser = argparse.ArgumentParser(program_name, *args, **kwargs)

    with importlib.resources.path(CRITERIA.resources, "CRITERIA.toml") as config_path:
        parser.add_argument(
            "--config",
            "-c",
            help="The configuration file.",
            type=str,
            default=config_path.__str__(),
        )
    return parser


def get_configuration(argparse_args: argparse.Namespace) -> dict:
    with open(argparse_args.config, "r") as fp:
        config = toml.load(fp)
    return config


def get_model_metrics_results(config, program_config):
    metrics_results_dir = config["metrics_results_dir"]
    file_formatter = get_file_formatter(config["file_formatter"])
    metrics_results_file = (
        program_config.get("metrics_results_file") or config["metrics_results_file"]
    ).format(metrics_results_dir=metrics_results_dir, ext=file_formatter.ext)
    with open(
        metrics_results_file,  # "metrics/models_metrics_results.pkl",
        "rb",
    ) as handle:
        models_metrics_results = file_formatter.formatter.load(handle)
    return models_metrics_results


def get_file_formatter(file_type: str) -> FileFormatter:
    if file_type == "json":
        import json

        return FileFormatter("json", json, ext="json", mode="t")
    elif file_type == "pickle" or file_type == "":
        return FileFormatter("pickle", pkl, ext="pkl", mode="b")
    elif file_type == "compress_pickle":
        import compress_pickle

        return FileFormatter("compress_pickle", compress_pickle, ext="pkl", mode="b")
    raise NotImplementedError


def load_source(model: Union[str, dict]):
    if isinstance(model, dict):
        model_source: str = model["source"]
    else:
        model_source = model
    with open(model_source, "rb") as fp:
        return pkl.load(fp)


def load_sources(model_dict: dict):
    """See [prediction_models] in `CRITERIA.toml`."""
    output = {}
    for model_name, source in model_dict.items():
        output[model_name] = load_source(source)
    return output


def get_translation(df, reference_frame=REFERENCE_FRAME):
    # Assign Frames to Timestamps
    ts_list = df["TIMESTAMP"].unique()
    ts_list.sort()

    ts_mask = []
    frames = []
    for i, ts in enumerate(ts_list):
        ts_mask.append(df["TIMESTAMP"] == ts)
        frames.append(i)

    df.loc[:, "FRAME"] = np.select(ts_mask, frames)

    # Get coordinate for AoI as translation.
    agent_df = df[df.OBJECT_TYPE == "AGENT"]  # Agent of Interest (AoI)
    translation = agent_df[agent_df.FRAME == reference_frame][["X", "Y"]].to_numpy()

    return translation.squeeze()


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
