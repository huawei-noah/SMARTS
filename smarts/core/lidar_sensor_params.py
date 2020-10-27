# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorParams:
    start_angle: float
    end_angle: float
    laser_angles: list
    angle_resolution: float
    max_distance: float
    noise_mu: float
    noise_sigma: float


VelodyneHDL32E = SensorParams(
    start_angle=0,
    end_angle=2 * np.pi,
    laser_angles=np.linspace(-np.radians(30.67), np.radians(10.67), 24),
    angle_resolution=0.1728,
    max_distance=100,
    noise_mu=0,
    noise_sigma=0.078,
)

BasicLidar = SensorParams(
    start_angle=0,
    end_angle=2 * np.pi,
    laser_angles=np.linspace(-np.radians(4), np.radians(10), 50),
    angle_resolution=1,
    max_distance=20,
    noise_mu=0,
    noise_sigma=0.078,
)
