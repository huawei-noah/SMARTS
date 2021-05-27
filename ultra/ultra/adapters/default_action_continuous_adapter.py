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
import gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


# The space of the adapted action.
gym_space: gym.Space = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32,
)
# This adapter reqiures SMARTS to ensure that the agent is provided a "continuous"
# controller, that is, a controller that allows for actions in the form of an array:
# [throttle, brake, steering].
required_interface = {"action": ActionSpaceType.Continuous}


def adapt(action: np.ndarray) -> np.ndarray:
    """Adapts a given action into an action that SMARTS can understand for a continuous
    controller. This adapter expects that the action is already a valid continuous
    controller action.

    Args:
        action (numpy.ndarray): The action to adapt. The action should be in the form of
            [throttle, brake, steering] where each element is a float. The throttle
            element is in the range [0, 1], the brake element is in the range [0, 1] and
            the steering element is in the range [-1, 1].

    Returns:
        np.ndarray: The same action that was passed in.
    """
    return action
