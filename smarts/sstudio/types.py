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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import re
import math
import random
import itertools
import collections.abc as collections_abc
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Dict, Any, Union, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from smarts.core import gen_id
from smarts.core.utils.id import SocialAgentId
from smarts.core.waypoints import Waypoint, Waypoints
from smarts.core.sumo_road_network import SumoRoadNetwork


class _SumoParams(collections_abc.Mapping):
    """For some Sumo params (e.x. LaneChangingModel) the arguments are in title case
    with a given prefix. Subclassing this class allows for an automatic way to map
    between PEP8-compatible naming and Sumo's.
    """

    def __init__(self, prefix, whitelist=[], **kwargs):
        def snake_to_title(word):
            return "".join(x.capitalize() or "_" for x in word.split("_"))

        # XXX: On rare occasions sumo doesn't respect their own conventions
        #      (e.x. junction model's impatience).
        self._params = {key: kwargs.pop(key) for key in whitelist if key in kwargs}

        for key, value in kwargs.items():
            self._params[f"{prefix}{snake_to_title(key)}"] = value

    def __iter__(self):
        return iter(self._params)

    def __getitem__(self, key):
        return self._params[key]

    def __len__(self):
        return len(self._params)

    def __hash__(self):
        return hash(frozenset(self._params.items()))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


class LaneChangingModel(_SumoParams):
    """Models how the actor acts with respect to lane changes."""

    def __init__(self, **kwargs):
        super().__init__("lc", **kwargs)


class JunctionModel(_SumoParams):
    """Models how the actor acts with respect to waiting at junctions."""

    def __init__(self, **kwargs):
        super().__init__("jm", whitelist=["impatience"], **kwargs)


@dataclass(frozen=True)
class Distribution:
    """A gaussian distribution used for randomized parameters."""

    mean: float
    """The mean value of the gaussian distribution."""
    sigma: float
    """The sigma value of the gaussian distribution."""

    def sample(self):
        """The next sample from the distribution."""
        return random.gauss(self.mean, self.sigma)


@dataclass
class UniformDistribution:
    """A uniform distribution, return a random number N
    such that a <= N <= b for a <= b and b <= N <= a for b < a.
    """

    a: float
    b: float

    def __post_init__(self):
        if self.b < self.a:
            self.a, self.b = self.b, self.a

    def sample(self):
        return random.uniform(self.a, self.b)


@dataclass
class TruncatedDistribution:
    """A truncated normal distribution, by default, location=0, scale=1"""

    a: float
    b: float
    loc: float = 0
    scale: float = 1

    def __post_init__(self):
        assert self.a != self.b
        if self.b < self.a:
            self.a, self.b = self.b, self.a

    def sample(self):
        from scipy.stats import truncnorm

        return truncnorm.rvs(self.a, self.b, loc=self.loc, scale=self.scale)


@dataclass(frozen=True)
class Actor:
    """This is the base description/spec type for traffic actors."""

    pass


@dataclass(frozen=True, unsafe_hash=True)
class TrafficActor(Actor):
    """Used as a description/spec for traffic actors (e.x. Vehicles, Pedestrians,
    etc). The defaults provided are for a car, but the name is not set to make it
    explicit that you actually want a car.
    """

    name: str
    """The name of the traffic actor. It must be unique."""
    accel: float = 2.6
    """The acceleration value of the actor."""
    decel: float = 4.5
    """The deceleration value of the actor."""
    tau: float = 1.0
    """The minimum time headway"""
    sigma: float = 0.5
    """The driver imperfection"""
    depart_speed: Union[float, str] = "max"
    """The starting speed of the actor"""
    emergency_decel: float = 4.5
    """maximum deceleration ability of vehicle in case of emergency"""
    speed: Distribution = Distribution(mean=1.0, sigma=0.1)
    """The speed distribution of this actor in m/s."""
    imperfection: Distribution = Distribution(mean=0.5, sigma=0)
    """Imperfection within range [0..1]"""
    min_gap: Distribution = Distribution(mean=2.5, sigma=0)
    """Minimum gap in meters."""
    vehicle_type: str = "passenger"
    """The type of vehicle this actor uses. ("passenger", "bus", "coach", "truck", "trailer")"""
    lane_changing_model: LaneChangingModel = field(
        default_factory=LaneChangingModel, hash=False
    )
    junction_model: JunctionModel = field(default_factory=JunctionModel, hash=False)

    @property
    def id(self) -> str:
        """The identifier tag of the traffic actor."""
        return "actor-{}-{}".format(self.name, hash(self))


@dataclass(frozen=True)
class SocialAgentActor(Actor):
    """Used as a description/spec for zoo traffic actors. These actors use a
    pre-trained model to understand how to act in the environment.
    """

    name: str
    """The name of the social actor. Must be unique."""

    # A pre-registered zoo identifying tag you provide to help SMARTS identify the
    # prefab of a social agent.
    agent_locator: str
    """The locator reference to the zoo registration call. Expects a string in the format
    of ‘path.to.file:locator-name’ where the path to the registration call is in the form
    {PYTHONPATH}[n]/path/to/file.py
    """
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to be passed to the constructed class overriding the
    existing registered arguments.
    """
    initial_speed: float = None
    """Set the initial speed, defaults to 0."""


@dataclass(frozen=True)
class BoidAgentActor(SocialAgentActor):
    """Used as a description/spec for boid traffic actors. Boid actors control multiple
    vehicles.
    """

    id: str = field(default_factory=lambda: f"boid-{gen_id()}")

    # The max number of vehicles that this agent will control at a time. This value is
    # honored when using a bubble for boid dynamic assignment.
    capacity: int = None
    """The capacity of the boid agent to take over vehicles."""


@dataclass(frozen=True)
class Route:
    """A route is represented by begin and end edge IDs, with an optional list of
    itermediary edge IDs. When an intermediary is not specified the router will
    decide what it should be.
    """

    ## edge, lane index, offset
    begin: Tuple[str, int, Any]
    """The (edge, lane_index, offset) details of the start location for the route.

    edge:
        The starting edge by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: "max", "random"
    """
    ## edge, lane index, offset
    end: Tuple[str, int, Any]
    """The (edge, lane_index, offset) details of the end location for the route.

    edge:
        The starting edge by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: "max", "random"
    """

    # Edges we want to make sure this route includes
    via: Tuple[str, ...] = field(default_factory=tuple)
    """The ids of edges that must be included in the route between `begin` and `end`."""

    @property
    def id(self) -> str:
        return "route-{}-{}-{}-".format(
            "_".join(map(str, self.begin)), "_".join(map(str, self.end)), hash(self),
        )

    @property
    def edges(self):
        return (self.begin[0],) + self.via + (self.end[0],)


@dataclass(frozen=True)
class RandomRoute:
    """An alternative to types.Route which specifies to sstudio to generate a random
    route.
    """

    id: str = field(default_factory=lambda: f"random-route-{gen_id()}")


@dataclass(frozen=True)
class Flow:
    """A route with an actor type emitted at a given rate."""

    route: Route
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
    """An actor to weight mapping associated as\\: { actor: weight }
    actor:
        The traffic actors that are provided.
    weight:
        The chance of this actor appearing as a ratio over total weight.
    """

    @property
    def id(self) -> str:
        return "flow-{}-{}-".format(
            self.route.id, str(hash(frozenset(self.actors.items())))
        )

    def __hash__(self):
        # Custom hash since self.actors is not hashable, here we first convert to a
        # frozenset.
        return hash((self.route, self.rate, frozenset(self.actors.items())))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


@dataclass(frozen=True)
class Traffic:
    """The descriptor for traffic."""

    flows: Sequence[Flow]
    """Flows are used to define a steady supply of vehicles."""


@dataclass(frozen=True)
class EntryTactic:
    pass


@dataclass(frozen=True)
class TrapEntryTactic(EntryTactic):
    """An entry tactic that hijacks a vehicle to start the mission."""

    wait_to_hijack_limit_s: float
    """The amount of seconds a hijack will wait to get a vehicle before just emitting"""
    zone: "MapZone" = None
    """The zone of the hijack area"""
    exclusion_prefixes: Tuple[str, ...] = tuple()
    """The prefixes of vehicles to avoid hijacking"""
    default_entry_speed: float = None
    """The speed that the vehicle starts at when defaulting to emitting"""


@dataclass(frozen=True)
class Mission:
    """The descriptor for an actor's mission."""

    route: Route
    """The route for the actor to attempt to follow."""
    start_time: float = 0.1
    """The earliest simulation time that this mission starts but may start later in couple with
    `entry_tactic`.
    """
    entry_tactic: EntryTactic = None
    """A specific tactic the mission should employ to start the mission."""


@dataclass(frozen=True)
class EndlessMission:
    """The descriptor for an actor's mission that has no end."""

    begin: Tuple[str, int, float]
    """The (edge, lane_index, offset) details of the start location for the route.

    edge:
        The starting edge by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: 'max', 'random'
    """
    start_time: float = 0.1
    """The earliest simulation time that this mission starts"""
    entry_tactic: EntryTactic = None
    """A specific tactic the mission should employ to start the mission"""


@dataclass(frozen=True)
class LapMission:
    """The descriptor for an actor's mission that defines mission that repeats
    in a closed loop.
    """

    route: Route
    """The route for the actor to attempt to follow"""
    num_laps: int
    """The amount of times to repeat the mission"""
    start_time: float = 0.1
    """The earliest simulation time that this mission starts"""
    entry_tactic: EntryTactic = None
    """A specific tactic the mission should employ to start the mission"""


@dataclass(frozen=True)
class GroupedLapMission:
    """The descriptor for a group of actor missions that repeat in a closed loop."""

    route: Route
    """The route for the actors to attempt to follow"""
    offset: int
    """The offset of the "starting line" for the group"""
    lanes: int
    """The number of lanes the group occupies"""
    actor_count: int
    """The number of actors to be part of the group"""
    num_laps: int
    """The amount of times to repeat the mission"""


@dataclass(frozen=True)
class Zone:
    """The base for a descriptor that defines a capture area."""

    def to_geometry(self, road_network: SumoRoadNetwork) -> Polygon:
        """Generates the geometry from this zone."""
        raise NotImplementedError


@dataclass(frozen=True)
class MapZone(Zone):
    """A descriptor that defines a capture area."""

    start: Tuple[str, int, float]
    """The (edge, lane_index, offset) details of the starting location.

    edge:
        The starting edge by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: 'max', 'random'
    """
    length: float
    """The length of the geometry along the center of the lane. Also acceptable\\: 'max'"""
    n_lanes: 2
    """The number of lanes from right to left that this zone covers."""

    def to_geometry(self, road_network: SumoRoadNetwork) -> Polygon:
        def resolve_offset(offset, geometry_length, lane_length, buffer_from_ends):
            if offset == "base" or offset == 0:
                return buffer_from_ends
            # push off of end of lane
            elif offset == "max":
                return lane_length - geometry_length - buffer_from_ends
            elif offset == "random":
                return random.uniform(
                    0, lane_length - geometry_length - buffer_from_ends
                )
            else:
                return float(offset)

        lane_shapes = []
        edge_id, lane_idx, offset = self.start
        edge = road_network.edge_by_id(edge_id)
        for lane_idx in range(lane_idx, lane_idx + self.n_lanes):
            lane = edge.getLanes()[lane_idx]
            lane_length = lane.getLength()
            geom_length = max(self.length - 1e-6, 1e-6)

            assert lane_length > geom_length  # Geom is too long for lane
            assert geom_length > 0  # Geom length is negative

            lane_shape = SumoRoadNetwork.buffered_lane_or_edge(
                lane, width=lane.getWidth() + 0.3
            )

            min_cut = resolve_offset(offset, geom_length, lane_length, 1e-6)
            # Second cut takes into account shortening of geometry by `min_cut`.
            max_cut = min(min_cut + geom_length, lane_length - min_cut - 1e-6)

            lane_shape = road_network.split_lane_shape_at_offset(
                Polygon(lane_shape), lane, min_cut
            )

            if isinstance(lane_shape, GeometryCollection):
                if len(lane_shape) < 2:
                    break
                lane_shape = lane_shape[1]

            lane_shape = road_network.split_lane_shape_at_offset(
                lane_shape, lane, max_cut,
            )[0]
            lane_shapes.append(lane_shape)

        geom = unary_union(MultiPolygon(lane_shapes))
        return geom


@dataclass(frozen=True)
class PositionalZone(Zone):
    """A descriptor that defines a capture area at a specific XY location."""

    # center point
    pos: Tuple[float, float]
    """A (x,y) position of the zone in the scenario."""
    size: Tuple[float, float]
    """The (length, width) dimensions of the zone."""

    def to_geometry(self, road_network: SumoRoadNetwork = None) -> Polygon:
        w, h = self.size
        p0 = (self.pos[0] - w / 2, self.pos[1] - h / 2)  # min
        p1 = (self.pos[0] + w / 2, self.pos[1] + h / 2)  # max
        return Polygon([p0, (p0[0], p1[1]), p1, (p1[0], p0[1])])


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
    limit: int = None
    """The maximum number of actors that could be captured."""
    exclusion_prefixes: Tuple[str, ...] = field(default_factory=tuple)
    """Used to exclude social actors from capture."""
    id: str = field(default_factory=lambda: f"bubble-{gen_id()}")
    follow_actor_id: str = None
    """Actor ID of agent we want to pin to. Doing so makes this a "travelling bubble"
    which means it moves to follow the `follow_actor_id`'s vehicle. Offset is from the
    vehicle's center position to the bubble's center position.
    """
    follow_offset: Tuple[float, float] = None
    """Maintained offset to place the travelling bubble relative to the follow
    vehicle if it were facing north.
    """
    keep_alive: bool = False
    """If enabled, the social agent actor will be spawned upon first vehicle airlock
    and be reused for every subsequent vehicle entering the bubble until the episode
    is over.
    """

    def __post_init__(self):
        if self.margin <= 0:
            raise ValueError("Airlocking margin must be greater than 0")

        if self.follow_actor_id is not None and self.follow_offset is None:
            raise ValueError(
                "A follow offset must be set if this is a travelling bubble"
            )

        if self.keep_alive and not self.is_boid:
            # TODO: We may want to remove this restriction in the future
            raise ValueError(
                "Only boids can have keep_alive enabled (for persistent boids)"
            )

    @staticmethod
    def to_actor_id(actor, mission_group):
        return SocialAgentId.new(actor.name, group=mission_group)

    @property
    def is_boid(self):
        return isinstance(self.actor, BoidAgentActor)


@dataclass(frozen=True)
class RoadSurfacePatch:
    """A descriptor that defines a patch of road surface with a different friction coefficent."""

    zone: Zone
    """The zone which to capture vehicles."""
    begin_time: int
    """The start time in seconds of when this surface is active."""
    end_time: int
    """The end time in seconds of when this surface is active."""
    friction_coefficient: float
    """The surface friction coefficient."""


@dataclass(frozen=True)
class _ActorAndMission:
    actor: Actor
    mission: Union[Mission, EndlessMission, LapMission]


@dataclass(frozen=True)
class Scenario:
    traffic: Optional[Dict[str, Traffic]] = None
    ego_missions: Optional[Sequence[Mission]] = None
    # e.g. { "turning_agents": ([actors], [missions]), ... }
    social_agent_missions: Optional[
        Dict[str, Tuple[Sequence[SocialAgentActor], Sequence[Mission]]]
    ] = None
    """Every dictionary item {group: (actors, missions)} gets run simultaneously. If
    actors > 1 and missions = 0 or actors = 1 and missions > 0 we cycle through them
    every episode. Otherwise actors must be the same length as missions.
    """
    bubbles: Optional[Sequence[Bubble]] = None
    friction_maps: Optional[Sequence[RoadSurfacePatch]] = None
    traffic_histories: Optional[Sequence[str]] = None
