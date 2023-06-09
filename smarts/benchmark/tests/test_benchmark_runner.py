# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
from unittest import mock

import pytest

from smarts.benchmark.driving_smarts import load_config
from smarts.benchmark.entrypoints.benchmark_runner_v0 import benchmark
from smarts.core.controllers import ActionSpaceType


@pytest.fixture(scope="module")
def get_benchmark_args(request):
    config_path = Path(__file__).resolve().parents[3] / request.param
    benchmark_args = load_config(config_path)["benchmark"]
    assert isinstance(benchmark_args, dict), f"Config path not found: {config_path}."
    benchmark_args.update({"eval_episodes": 2})
    return benchmark_args


@pytest.mark.parametrize(
    "get_benchmark_args",
    [
        "smarts/benchmark/driving_smarts/v2023/config_1.yaml",
        "smarts/benchmark/driving_smarts/v2023/config_2.yaml",
        "smarts/benchmark/driving_smarts/v2023/config_3.yaml",
    ],
    indirect=True,
)
@mock.patch(
    "smarts.env.gymnasium.platoon_env.SUPPORTED_ACTION_TYPES",
    (ActionSpaceType.LaneWithContinuousSpeed, ActionSpaceType.Lane),
)
@mock.patch(
    "smarts.env.gymnasium.driving_smarts_2023_env.SUPPORTED_ACTION_TYPES",
    (ActionSpaceType.LaneWithContinuousSpeed, ActionSpaceType.Lane),
)
def test_benchmark(get_benchmark_args):
    agent_locator = "zoo.policies:keep-lane-agent-v0"

    # Verify that benchmark runs without errors.
    benchmark(benchmark_args=get_benchmark_args, agent_locator=agent_locator)
