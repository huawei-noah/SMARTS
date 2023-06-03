# MIT License

# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    """Performance cost values."""

    collisions: int = 0
    comfort: float = 0
    dist_to_destination: float = 0
    dist_to_obstacles: float = 0
    jerk_linear: float = 0
    lane_center_offset: float = 0
    off_road: int = 0
    speed_limit: float = 0
    steps: float = 0
    vehicle_gap: float = 0
    wrong_way: float = 0


@dataclass(frozen=True)
class Counts:
    """Performance count values."""

    goals: int = 0
    """ Number of episodes completed successfully by achieving the goal.
    """
    episodes: int = 0
    """ Number of episodes traversed.
    """
    steps: int = 0
    """ Sum of steps taken over all episodes.
    """


@dataclass(frozen=True)
class Metadata:
    """Metadata of the record."""

    difficulty: float = 1
    """Task difficulty value.
    """


@dataclass
class Record:
    """Stores an agent's performance-cost, performance-count, and
    performance-metadata values."""

    costs: Costs
    counts: Counts
    metadata: Metadata
