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


from dataclasses import dataclass
from typing import Optional, Tuple

from smarts.core.condition_state import ConditionState
from smarts.sstudio.sstypes.condition import (
    Condition,
    ConditionRequires,
    LiteralCondition,
)
from smarts.sstudio.sstypes.zone import MapZone


@dataclass(frozen=True)
class EntryTactic:
    """The tactic that the simulation should use to acquire a vehicle for an agent."""

    start_time: float

    def __post_init__(self):
        assert (
            getattr(self, "condition", None) is not None
        ), "Abstract class, inheriting types must implement the `condition` field."


@dataclass(frozen=True)
class TrapEntryTactic(EntryTactic):
    """An entry tactic that repurposes a pre-existing vehicle for an agent."""

    wait_to_hijack_limit_s: float = 0
    """The amount of seconds a hijack will wait to get a vehicle before defaulting to a new vehicle"""
    zone: Optional[MapZone] = None
    """The zone of the hijack area"""
    exclusion_prefixes: Tuple[str, ...] = tuple()
    """The prefixes of vehicles to avoid hijacking"""
    default_entry_speed: Optional[float] = None
    """The speed that the vehicle starts at when the hijack limit expiry emits a new vehicle"""
    condition: Condition = LiteralCondition(ConditionState.TRUE)
    """A condition that is used to add additional exclusions."""

    def __post_init__(self):
        assert isinstance(self.condition, (Condition))
        assert not (
            self.condition.requires & ConditionRequires.any_current_actor_state
        ), f"Trap entry tactic cannot use conditions that require any_vehicle_state."


@dataclass(frozen=True)
class IdEntryTactic(EntryTactic):
    """An entry tactic which repurposes a pre-existing actor for an agent. Selects that actor by id."""

    actor_id: str
    """The id of the actor to take over."""

    condition: Condition = LiteralCondition(ConditionState.TRUE)
    """A condition that is used to add additional exclusions."""

    def __post_init__(self):
        assert isinstance(self.actor_id, str)
        assert isinstance(self.condition, (Condition))
