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
from enum import IntEnum
from typing import Optional


class ActorRole(IntEnum):
    """Used to specify the role an actor (e.g. vehicle) is currently playing in the simulation."""

    Unknown = 0

    # Vehicle Roles
    Social = 1  # Traffic
    SocialAgent = 2
    EgoAgent = 3

    # Non-vehicle Roles
    Signal = 4

    # For deferring to external co-simulators only. Cannot be hijacked or trapped.
    # Privileged state, so use with caution!
    External = 5


@dataclass
class ActorState:
    """Actor state information."""

    actor_id: str  # must be unique within the simulation
    actor_type: Optional[str] = None
    source: Optional[str] = None  # the source of truth for this Actor's state
    role: ActorRole = ActorRole.Unknown
    updated: bool = False

    def __lt__(self, other) -> bool:
        """Allows ordering ActorStates for use in sorted data-structures."""
        assert isinstance(other, ActorState)
        return self.actor_id < other.actor_id or (
            self.actor_id == other.actor_id and id(self) < id(other)
        )

    def __hash__(self) -> int:
        # actor_id must be unique within the simulation
        return hash(self.actor_id)

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and hash(self) == hash(other)
