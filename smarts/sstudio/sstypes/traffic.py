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

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Union

from smarts.core.utils.file import pickle_hash_int
from smarts.sstudio.sstypes.actor.traffic_actor import TrafficActor
from smarts.sstudio.sstypes.route import RandomRoute, Route


@dataclass(frozen=True)
class Flow:
    """A route with an actor type emitted at a given rate."""

    route: Union[RandomRoute, Route]
    """The route for the actor to attempt to follow."""
    rate: float
    """Vehicles per hour."""
    begin: float = 0
    """Start time in seconds."""
    # XXX: Defaults to 1 hour of traffic. We may want to change this to be "continual
    #      traffic", effectively an infinite end.
    end: float = 1 * 60 * 60
    """End time in seconds."""
    actors: Dict[TrafficActor, float] = field(default_factory=dict)
    """An actor to weight mapping associated as { actor -> weight }.

    :param actor: The traffic actors that are provided.
    :param weight: The chance of this actor appearing as a ratio over total weight.
    """
    randomly_spaced: bool = False
    """Determines if the flow should have randomly spaced traffic. Defaults to `False`."""
    repeat_route: bool = False
    """If True, vehicles that finish their route will be restarted at the beginning. Defaults to `False`."""

    @property
    def id(self) -> str:
        """The unique id of this flow."""
        return "{}-{}".format(
            self.route.id,
            str(hash(self))[:6],
        )

    def __hash__(self):
        # Custom hash since self.actors is not hashable, here we first convert to a
        # frozenset.
        return pickle_hash_int((self.route, self.rate, frozenset(self.actors.items())))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


@dataclass(frozen=True)
class Trip:
    """A route with a single actor type with name and unique id."""

    vehicle_name: str
    """The name of the vehicle. It must be unique. """
    route: Union[RandomRoute, Route]
    """The route for the actor to attempt to follow."""
    vehicle_type: str = "passenger"
    """The type of the vehicle"""
    depart: float = 0
    """Start time in seconds."""
    actor: Optional[TrafficActor] = field(default=None)
    """The traffic actor model (usually vehicle) that will be used for the trip."""

    def __post_init__(self):
        object.__setattr__(
            self,
            "actor",
            (
                replace(
                    self.actor, name=self.vehicle_name, vehicle_type=self.vehicle_type
                )
                if self.actor is not None
                else TrafficActor(
                    name=self.vehicle_name, vehicle_type=self.vehicle_type
                )
            ),
        )

    @property
    def id(self) -> str:
        """The unique id of this trip."""
        return self.vehicle_name

    def __hash__(self):
        # Custom hash since self.actors is not hashable, here we first convert to a
        # frozenset.
        return pickle_hash_int((self.route, self.actor))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


@dataclass(frozen=True)
class Traffic:
    """The descriptor for traffic."""

    flows: Sequence[Flow]
    """Flows are used to define a steady supply of vehicles."""
    # TODO: consider moving TrafficHistory stuff in here (and rename to Trajectory)
    # TODO:  - treat history points like Vias (no guarantee on history timesteps anyway)
    trips: Optional[Sequence[Trip]] = None
    """Trips are used to define a series of single vehicle trip."""
    engine: str = "SUMO"
    """Traffic-generation engine to use. Supported values include "SUMO" and "SMARTS". "SUMO" requires using a SumoRoadNetwork for the RoadMap.
    """
