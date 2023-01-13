# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

from dataclasses import dataclass


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
    steps_adjusted: int = 0
    """ Sum of steps taken over all episodes. The number of steps, in an episode
    where the agent did not achieve the goal, is replaced with the agent's  
    max_episode_steps value.
    """
    max_steps: int = 0
    """ Sum of maximum number of steps over all episodes.
    """
