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
from typing import Optional, Tuple

from smarts.core import gen_id
from smarts.core.condition_state import ConditionState
from smarts.core.utils.id import SocialAgentId
from smarts.sstudio.types.actor.social_agent_actor import (
    BoidAgentActor,
    SocialAgentActor,
)
from smarts.sstudio.types.bubble_limits import BubbleLimits
from smarts.sstudio.types.condition import Condition, LiteralCondition
from smarts.sstudio.types.zone import MapZone, Zone


@dataclass(frozen=True)
class Bubble:
    """A descriptor that defines a capture bubble for social agents."""

    zone: Zone
    """The zone which to capture vehicles."""
    actor: SocialAgentActor
    """The actor specification that this bubble works for."""
    margin: float = 2  # Used for "airlocking"; must be > 0
    """The exterior buffer area for airlocking. Must be > 0."""
    # If limit != None it will only allow that specified number of vehicles to be
    # hijacked. N.B. when actor = BoidAgentActor the lesser of the actor capacity
    # and bubble limit will be used.
    limit: Optional[BubbleLimits] = None
    """The maximum number of actors that could be captured."""
    exclusion_prefixes: Tuple[str, ...] = field(default_factory=tuple)
    """Used to exclude social actors from capture."""
    id: str = field(default_factory=lambda: f"bubble-{gen_id()}")
    follow_actor_id: Optional[str] = None
    """Actor ID of agent we want to pin to. Doing so makes this a "travelling bubble"
    which means it moves to follow the `follow_actor_id`'s vehicle. Offset is from the
    vehicle's center position to the bubble's center position.
    """
    follow_offset: Optional[Tuple[float, float]] = None
    """Maintained offset to place the travelling bubble relative to the follow
    vehicle if it were facing north.
    """
    keep_alive: bool = False
    """If enabled, the social agent actor will be spawned upon first vehicle airlock
    and be reused for every subsequent vehicle entering the bubble until the episode
    is over.
    """
    follow_vehicle_id: Optional[str] = None
    """Vehicle ID of a vehicle we want to pin to. Doing so makes this a "travelling bubble"
    which means it moves to follow the `follow_vehicle_id`'s vehicle. Offset is from the
    vehicle's center position to the bubble's center position.
    """
    condition: Condition = LiteralCondition(ConditionState.TRUE)

    def __post_init__(self):
        if self.margin < 0:
            raise ValueError("Airlocking margin must be greater than 0")

        if self.follow_actor_id is not None and self.follow_vehicle_id is not None:
            raise ValueError(
                "Only one option of follow actor id and follow vehicle id can be used at any time."
            )

        if (
            self.follow_actor_id is not None or self.follow_vehicle_id is not None
        ) and self.follow_offset is None:
            raise ValueError(
                "A follow offset must be set if this is a travelling bubble"
            )

        if self.keep_alive and not self.is_boid:
            # TODO: We may want to remove this restriction in the future
            raise ValueError(
                "Only boids can have keep_alive enabled (for persistent boids)"
            )

        if not isinstance(self.zone, MapZone):
            poly = self.zone.to_geometry(road_map=None)
            if not poly.is_valid:
                follow_id = (
                    self.follow_actor_id
                    if self.follow_actor_id
                    else self.follow_vehicle_id
                )
                raise ValueError(
                    f"The zone polygon of {type(self.zone).__name__} of moving {self.id} which following {follow_id} is not a valid closed loop"
                    if follow_id
                    else f"The zone polygon of {type(self.zone).__name__} of fixed position {self.id} is not a valid closed loop"
                )

    @staticmethod
    def to_actor_id(actor, mission_group):
        """Mashes the actor id and mission group to create what needs to be a unique id."""
        return SocialAgentId.new(actor.name, group=mission_group)

    @property
    def is_boid(self):
        """Tests if the actor is to control multiple vehicles."""
        return isinstance(self.actor, BoidAgentActor)
