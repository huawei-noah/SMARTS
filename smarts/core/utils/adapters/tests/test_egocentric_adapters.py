# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONmath.infRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np
import pytest

from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.core.utils.adapters.ego_centric_adapters import (
    ego_centric_observation_adapter,
    get_egocentric_adapters,
)
from smarts.core.utils.tests.fixtures import adapter_data, large_observation


def test_egocentric_observation_adapter(large_observation: Observation):
    new_obs: Observation = ego_centric_observation_adapter(large_observation)
    assert not np.allclose(
        large_observation.ego_vehicle_state.position, new_obs.ego_vehicle_state.position
    )
    assert (
        large_observation.ego_vehicle_state.heading != new_obs.ego_vehicle_state.heading
    )


def _is_same(v1, v2):
    if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        return np.allclose(v1, v2)
    elif isinstance(v1, (list, tuple)):
        return np.allclose(np.asarray(v1), np.asarray(v2))
    return v1 == v2


def test_adapters(adapter_data, large_observation: Observation):
    for action_space_type, action, expected_action in adapter_data:
        obs_adapter, act_adapter = get_egocentric_adapters(action_space_type)

        _ = obs_adapter(large_observation)
        augmented_action = act_adapter(action)
        print(f"{augmented_action} vs expected {expected_action}")
        assert _is_same(
            augmented_action, expected_action
        ), f"Type: {action_space_type}, base_obs: {large_observation.ego_vehicle_state}"
