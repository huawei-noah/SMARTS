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

from smarts.core.agent_interface import RGB
from smarts.core.sensors import Observation


# The width of the top-down RGB image.
_WIDTH = 64
# The height of the top-down RGB image.
_HEIGHT = 64
# The resolution of the top-down RGB image. E.g. A 100m x 100m RGB with a 256x256 image
# representation, needs a resolution of 100 / 256 and equal width and height of 256.
_RESOLUTION = 50 / 64


# The space of the adapted observation.
gym_space: gym.Space = gym.spaces.Box(
    low=0.0, high=1.0, shape=(_HEIGHT, _WIDTH), dtype=np.float32
)
# This adapter requires SMARTS to pass the top-down RGB image in the agent's
# observation.
required_interface = {"rgb": RGB(width=_WIDTH, height=_HEIGHT, resolution=_RESOLUTION)}


def adapt(observation: Observation) -> np.ndarray:
    """Adapts a raw environment observation into a numpy.ndarray.

    The raw observation from the environment must include the top down RGB image. See
    smarts.core.sensors for more information on the Observation type.

    Args:
        observation (Observation): The raw environment observation received from SMARTS.

    Returns:
        np.ndarray: A numpy.ndarray of size (64, 64) that is the gray-scale image of the
            top down RGB image. The gray-scale value for each pixel is calculated as
            0.2125 * r + 0.7154 * g + 0.0721 * b.
    """
    rgb_image = observation.top_down_rgb.data
    gray_image = np.dot(rgb_image, (0.2125, 0.7154, 0.0721))
    normalized_gray_image = np.divide(gray_image, 255.0)
    return normalized_gray_image.astype(np.float32)
