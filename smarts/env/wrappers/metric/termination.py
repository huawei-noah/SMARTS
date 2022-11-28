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

from enum import Enum


class Reason(Enum):
    """Agent termination status.
    """
    Goal = 0
    """Agent achieved its goal."""
    Crash = 1
    """ Agent becomes done due to collision, driving off road, or reaching max
    episode steps.
    """


def reason(obs) -> Reason:
    """Returns the agent's termination status.

    Args:
        obs (_type_): Agent's observation.

    Raises:
        Exception: Reason for agent termination unknown.

    Returns:
        Reason: Reason for agent termination.
    """
    if obs.events.reached_goal:
        return Reason.Goal
    elif (
        len(obs.events.collisions) > 0
        or obs.events.off_road
        or obs.events.reached_max_episode_steps
    ):
        return Reason.Crash
    else:
        raise Exception(f"Unsupported agent done reason. Events: {obs.events}.")
