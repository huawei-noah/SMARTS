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


from typing import Optional, Sequence, Tuple

import numpy as np

from smarts.core.coordinates import Dimensions, Pose


class ColliderBase:
    """The base collider"""

    @property
    def dimensions(self) -> Dimensions:
        """The fitted front aligned dimensions of the chassis."""
        raise NotImplementedError

    @property
    def contact_points(self) -> Sequence:
        """The contact point of the chassis."""
        raise NotImplementedError

    @property
    def speed(self) -> float:
        """The speed of the chassis in the facing direction of the chassis."""
        raise NotImplementedError

    @speed.setter
    def speed(self, speed: float):
        """Apply GCD from front-end."""
        raise NotImplementedError

    @property
    def pose(self) -> Pose:
        """The pose of the chassis."""
        raise NotImplementedError

    def set_pose(self, pose: Pose):
        """Use with caution since it disrupts the physics simulation. Sets the pose of the
        chassis."""
        raise NotImplementedError

    @property
    def velocity_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns linear velocity vector in m/s and angular velocity in rad/sec."""
        raise NotImplementedError

    @property
    def yaw_rate(self) -> float:
        """The turning rate of the chassis in radians."""
        raise NotImplementedError

    def step(self, current_simulation_time):
        """Update the chassis state."""
        raise NotImplementedError

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """Use with care!  In essence, this is tinkering with the physics of the world,
        and may have unintended behavioral or performance consequences."""
        raise NotImplementedError
