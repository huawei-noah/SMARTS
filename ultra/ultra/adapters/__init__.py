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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import enum
from typing import Any, Callable, Dict

import gym

import ultra.adapters.default_action_continuous_adapter as default_action_continuous_adapter
import ultra.adapters.default_action_discrete_adapter as default_action_discrete_adapter
import ultra.adapters.default_info_adapter as default_info_adapter
import ultra.adapters.default_observation_image_adapter as default_observation_image_adapter
import ultra.adapters.default_observation_vector_adapter as default_observation_vector_adapter
import ultra.adapters.default_reward_adapter as default_reward_adapter


class AdapterType(enum.Enum):
    # Action adapters.
    DefaultActionContinuous = enum.auto()
    DefaultActionDiscrete = enum.auto()

    # Info adapters.
    DefaultInfo = enum.auto()

    # Observation adapters.
    DefaultObservationVector = enum.auto()
    DefaultObservationImage = enum.auto()

    # Reward adapters.
    DefaultReward = enum.auto()


_STRING_TO_TYPE = {
    "default_action_continuous": AdapterType.DefaultActionContinuous,
    "default_action_discrete": AdapterType.DefaultActionDiscrete,
    "default_info": AdapterType.DefaultInfo,
    "default_observation_image": AdapterType.DefaultObservationImage,
    "default_observation_vector": AdapterType.DefaultObservationVector,
    "default_reward": AdapterType.DefaultReward,
}
_TYPE_TO_SPACE = {
    AdapterType.DefaultActionContinuous: default_action_continuous_adapter.gym_space,
    AdapterType.DefaultActionDiscrete: default_action_discrete_adapter.gym_space,
    AdapterType.DefaultObservationImage: default_observation_image_adapter.gym_space,
    AdapterType.DefaultObservationVector: default_observation_vector_adapter.gym_space,
}
_TYPE_TO_ADAPTER = {
    AdapterType.DefaultActionContinuous: default_action_continuous_adapter.adapt,
    AdapterType.DefaultActionDiscrete: default_action_discrete_adapter.adapt,
    AdapterType.DefaultInfo: default_info_adapter.adapt,
    AdapterType.DefaultObservationImage: default_observation_image_adapter.adapt,
    AdapterType.DefaultObservationVector: default_observation_vector_adapter.adapt,
    AdapterType.DefaultReward: default_reward_adapter.adapt,
}
_TYPE_TO_REQUIRED_INTERFACE = {
    AdapterType.DefaultActionContinuous: default_action_continuous_adapter.required_interface,
    AdapterType.DefaultActionDiscrete: default_action_discrete_adapter.required_interface,
    AdapterType.DefaultInfo: default_info_adapter.required_interface,
    AdapterType.DefaultObservationImage: default_observation_image_adapter.required_interface,
    AdapterType.DefaultObservationVector: default_observation_vector_adapter.required_interface,
    AdapterType.DefaultReward: default_reward_adapter.required_interface,
}


def type_from_string(string_type: str) -> AdapterType:
    """Returns the AdapterType of the given string.

    Args:
        string_type (str): The string corresponding to a unique adapter type.

    Returns:
        AdapterType: The respective AdapterType of the input string.

    Raises:
        Exception: If string_type has no AdapterType.
    """
    if string_type in _STRING_TO_TYPE:
        return _STRING_TO_TYPE[string_type]
    raise Exception(f"An adapter type is not set for string '{string_type}'.")


def space_from_type(adapter_type: AdapterType) -> gym.Space:
    """Returns the Gym space of the given AdapterType.

    Args:
        adapter_type (AdapterType): The AdapterType for the desired Gym space.

    Returns:
        gym.Space: The Gym space of adapter_type.

    Raises:
        Exception: If adapter_type has no Gym space.
    """
    if adapter_type in _TYPE_TO_SPACE:
        return _TYPE_TO_SPACE[adapter_type]
    raise Exception(f"A Gym Space is not set for adapter type {adapter_type}.")


def adapter_from_type(adapter_type: AdapterType) -> Callable:
    """Returns the adapter function of the given AdapterType.

    Args:
        adapter_type (AdapterType): The AdapterType for the desired adapter function.

    Returns:
        Callable: The adapter function of adapter_type.

    Raises:
        Exception: If adapter_type has no adapter function.
    """
    if adapter_type in _TYPE_TO_ADAPTER:
        return _TYPE_TO_ADAPTER[adapter_type]
    raise Exception(f"An adapter function is not set for adapter type {adapter_type}.")


def required_interface_from_types(*adapter_types: AdapterType) -> Dict[str, Any]:
    """Returns the union of the required interfaces for all given AdapterTypes.\

    If multiple given AdapterTypes require the same interface, the interface must be the
    same among all of all of them. For example, if AdapterType.MyFirstAdapter requires
    {"waypoints": Waypoints(20)} and AdapterType.MySecondAdapter requires
    {"waypoints": Waypoints(40)}, this function will raise an exception becaues the
    waypoints required interface is not the same among given AdapterTypes.

    Args:
        *adapter_types: A variable length argument list of AdapterTypes used to obtain
            the desired required interface union.

    Returns:
        dict: A dictionary containing the required interface.

    Raises:
        Exception: If an invalid AdapterType is given, or multiple AdapterTypes have the
            same interface requirement, but the arguments of that interface requirement
            differ.
    """
    required_interface = {}

    for adapter_type in adapter_types:
        if adapter_type not in _TYPE_TO_REQUIRED_INTERFACE:
            raise Exception(
                f"A required interface is not set for adapter type {adapter_type}."
            )
        adapter_type_interface = _TYPE_TO_REQUIRED_INTERFACE[adapter_type]

        # Ensure current interface requirements don't conflict with previous interface
        # requirements of other adapter types.
        for interface_name, interface_requirement in adapter_type_interface.items():
            if (
                interface_name in required_interface
                and required_interface[interface_name] != interface_requirement
            ):
                raise Exception(
                    f"Cannot resolve current {interface_requirement} requirement with "
                    f"existing {required_interface[interface_name]} requirement."
                )
            else:
                required_interface[interface_name] = interface_requirement

    return required_interface
