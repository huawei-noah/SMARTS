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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import NamedTuple, Sequence


class Events(NamedTuple):
    """Classified observations that can trigger agent done status."""

    collisions: Sequence  # Sequence[Collision]
    """Collisions with other vehicles (if any)."""
    off_road: bool
    """True if vehicle is off the road, else False."""
    off_route: bool
    """True if vehicle is off the mission route, else False."""
    on_shoulder: bool
    """True if vehicle goes on road shoulder, else False."""
    wrong_way: bool
    """True if vehicle is heading against the legal driving direction of the lane, else False."""
    not_moving: bool
    """True if vehicle has not moved for the configured amount of time, else False."""
    reached_goal: bool
    """True if vehicle has reached its mission goal, else False."""
    reached_max_episode_steps: bool
    """True if vehicle has reached its max episode steps, else False."""
    agents_alive_done: bool
    """True if all configured co-simulating agents are done (if any), else False. 
    This is useful for cases when the vehicle is related to other vehicles."""
