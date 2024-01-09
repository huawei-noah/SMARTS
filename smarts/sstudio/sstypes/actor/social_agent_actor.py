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


from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from smarts.core import gen_id
from smarts.sstudio.sstypes.actor import Actor
from smarts.sstudio.sstypes.bubble_limits import BubbleLimits


@dataclass(frozen=True)
class SocialAgentActor(Actor):
    """Used as a description/spec for zoo traffic actors. These actors use a
    pre-trained model to understand how to act in the environment.
    """

    # A pre-registered zoo identifying tag you provide to help SMARTS identify the
    # prefab of a social agent.
    agent_locator: str
    """The locator reference to the zoo registration call. Expects a string in the format
    of 'path.to.file:locator-name' where the path to the registration call is in the form
    `{PYTHONPATH}[n]/path/to/file.py`
    """
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to be passed to the constructed class overriding the
    existing registered arguments.
    """
    initial_speed: Optional[float] = None
    """Set the initial speed, defaults to 0."""


@dataclass(frozen=True)
class BoidAgentActor(SocialAgentActor):
    """Used as a description/spec for boid traffic actors. Boid actors control multiple
    vehicles.
    """

    id: str = field(default_factory=lambda: f"boid-{gen_id()}")

    # The max number of vehicles that this agent will control at a time. This value is
    # honored when using a bubble for boid dynamic assignment.
    capacity: Optional[BubbleLimits] = None
    """The capacity of the boid agent to take over vehicles."""
