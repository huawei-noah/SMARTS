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


@pytest.fixture(scope="module")
def get_benchmark_args(request):
    config_path = Path(__file__).resolve().parents[3] / request.param
    benchmark_args = load_config(config_path)
    assert isinstance(benchmark_args, dict), f"Config path not found: {config_path}."
    benchmark_args = benchmark_args["benchmark"]
    benchmark_args.update({"eval_episodes": 2})
    return benchmark_args


def _get_model(action):
    class MockModel:
        def predict(*args, **kwargs):
            return (action, None)

    return lambda _: MockModel()


@pytest.mark.parametrize(
    "get_benchmark_args",
    [
        "smarts/benchmark/driving_smarts/v2023/config_1.yaml",
        "smarts/benchmark/driving_smarts/v2023/config_2.yaml",
    ],
    indirect=True,
)
def test_drive(get_benchmark_args):
    """Tests Driving SMARTS 2023.1 and 2023.2 benchmarks using `examples/10_drive` model."""
    from contrib_policy.policy import Policy

    agent_locator = "examples.rl.drive.inference:contrib-agent-v0"
    action = 1
    with mock.patch.object(Policy, "_get_model", _get_model(action)):
        benchmark(benchmark_args=get_benchmark_args, agent_locator=agent_locator)


@pytest.mark.parametrize(
    "get_benchmark_args",
    [
        "smarts/benchmark/driving_smarts/v2023/config_3.yaml",
    ],
    indirect=True,
)
def test_platoon(get_benchmark_args):
    """Tests Driving SMARTS 2023.3 benchmark using `examples/11_platoon` model."""
    from contrib_policy.policy import Policy

    agent_locator = "examples.rl.platoon.inference:contrib-agent-v0"
    action = 2
    with mock.patch.object(Policy, "_get_model", _get_model(action)):
        benchmark(benchmark_args=get_benchmark_args, agent_locator=agent_locator)
