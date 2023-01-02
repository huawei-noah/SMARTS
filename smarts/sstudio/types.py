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
import collections.abc as collections_abc
import logging
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from sys import maxsize
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import split, unary_union

from smarts.core import gen_id
from smarts.core.coordinates import RefLinePoint
from smarts.core.default_map_builder import get_road_map
from smarts.core.road_map import RoadMap
from smarts.core.utils.file import pickle_hash_int
from smarts.core.utils.id import SocialAgentId
from smarts.core.utils.math import rotate_cw_around_point


class _SUMO_PARAMS_MODE(IntEnum):
    TITLE_CASE = 0
    KEEP_SNAKE_CASE = 1


class _SumoParams(collections_abc.Mapping):
    """For some Sumo params (e.g. LaneChangingModel) the arguments are in title case
    with a given prefix. Subclassing this class allows for an automatic way to map
    between PEP8-compatible naming and Sumo's.
    """

    def __init__(
        self, prefix, whitelist=[], mode=_SUMO_PARAMS_MODE.TITLE_CASE, **kwargs
    ):
        def snake_to_title(word):
            return "".join(x.capitalize() or "_" for x in word.split("_"))

        def keep_snake_case(word: str):
            w = word[0].upper() + word[1:]
            return "".join(x or "_" for x in w.split("_"))

        func: Callable[[str], str] = snake_to_title
        if mode == _SUMO_PARAMS_MODE.TITLE_CASE:
            pass
        elif mode == _SUMO_PARAMS_MODE.KEEP_SNAKE_CASE:
            func = keep_snake_case

        # XXX: On rare occasions sumo doesn't respect their own conventions
        #      (e.x. junction model's impatience).
        self._params = {key: kwargs.pop(key) for key in whitelist if key in kwargs}

        for key, value in kwargs.items():
            self._params[f"{prefix}{func(key)}"] = value

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

    # For SUMO-specific attributes, see:
    # https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models

    def __init__(self, **kwargs):
        super().__init__("lc", whitelist=["minGapLat"], **kwargs)


class JunctionModel(_SumoParams):
    """Models how the actor acts with respect to waiting at junctions."""

    def __init__(self, **kwargs):
        super().__init__("jm", whitelist=["impatience"], **kwargs)


class SmartsLaneChangingModel(LaneChangingModel):
    """Implements the simple lane-changing model built-into SMARTS.

    Args:
        cutin_prob (float, optional): Float value [0, 1] that
            determines the probabilty this vehicle will "arbitrarily" cut in
            front of an adjacent agent vehicle when it has a chance, even if
            there would otherwise be no reason to change lanes at that point.
            Higher values risk a situation where this vehicle ends up in a lane
            where it cannot maintain its planned route. If that happens, this
            vehicle will perform whatever its default behavior is when it
            completes its route. Defaults to 0.0.
        assertive (float, optional): Willingness to accept lower front and rear
            gaps in the target lane. The required gap is divided by this value.
            Attempts to match the semantics of the attribute in SUMO's default
            lane-changing model, see: ``https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models``.
            Range: positive reals. Defaults to 1.0.
        dogmatic (bool, optional): If True, will cutin when a suitable
            opportunity presents itself based on the above parameters, even if
            it means the risk of not not completing the assigned route;
            otherwise, will forego the chance. Defaults to True.
        hold_period (float, optional): The minimum amount of time (seconds) to
            remain in the agent's lane after cutting into it (including the
            time it takes within the lane to complete the maneuver). Must be
            non-negative. Defaults to 3.0.
        slow_down_after (float, optional): Target speed during the hold_period
            will be scaled by this value. Must be non-negative. Defaults to 1.0.
        multi_lane_cutin (bool, optional): If True, this vehicle will consider
            changing across multiple lanes at once in order to cutin upon an
            agent vehicle when there's an opportunity. Defaults to False.
    """

    def __init__(
        self,
        cutin_prob: float = 0.0,
        assertive: float = 1.0,
        dogmatic: bool = True,
        hold_period: float = 3.0,
        slow_down_after: float = 1.0,
        multi_lane_cutin: bool = False,
    ):
        super().__init__(
            cutin_prob=cutin_prob,
            assertive=assertive,
            dogmatic=dogmatic,
            hold_period=hold_period,
            slow_down_after=slow_down_after,
            multi_lane_cutin=multi_lane_cutin,
        )


class SmartsJunctionModel(JunctionModel):
    """Implements the simple junction model built-into SMARTS.

    Args:
        yield_to_agents (str, optional): Defaults to "normal". 3 options are
            available, namely: (1) "always" - Traffic actors will yield to Ego
            and Social agents within junctions. (2) "never" - Traffic actors
            will never yield to Ego or Social agents within junctions.
            (3) "normal" - Traffic actors will attempt to honor normal
            right-of-way conventions, only yielding when an agent has the
            right-of-way. Examples of such conventions include (a) vehicles
            going straight have the right-of-way over turning vehicles;
            (b) vehicles on roads with more lanes have the right-of-way
            relative to vehicles on intersecting roads with less lanes;
            (c) all other things being equal, the vehicle to the right
            in a counter-clockwise sense has the right-of-way.
        wait_to_restart (float, optional): The amount of time in seconds
            after stopping at a signal or stop sign before this vehicle
            will start to go again. Defaults to 0.0.
    """

    def __init__(self, yield_to_agents: str = "normal", wait_to_restart: float = 0.0):
        super().__init__(
            yield_to_agents=yield_to_agents, wait_to_restart=wait_to_restart
        )


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
        """Get the next sample."""
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
        """Get the next sample"""
        from scipy.stats import truncnorm

        return truncnorm.rvs(self.a, self.b, loc=self.loc, scale=self.scale)


@dataclass(frozen=True)
class Actor:
    """This is the base description/spec type for traffic actors."""

    pass


@dataclass(frozen=True)
class TrafficActor(Actor):
    """Used as a description/spec for traffic actors (e.x. Vehicles, Pedestrians,
    etc). The defaults provided are for a car, but the name is not set to make it
    explicit that you actually want a car.
    """

    name: str
    """The name of the traffic actor. It must be unique."""
    accel: float = 2.6
    """The maximum acceleration value of the actor (in m/s^2)."""
    decel: float = 4.5
    """The maximum deceleration value of the actor (in m/s^2)."""
    tau: float = 1.0
    """The minimum time headway"""
    sigma: float = 0.5
    """The driver imperfection"""  # TODO: appears to not be used in generators.py
    depart_speed: Union[float, str] = "max"
    """The starting speed of the actor"""
    emergency_decel: float = 4.5
    """maximum deceleration ability of vehicle in case of emergency"""
    speed: Distribution = Distribution(mean=1.0, sigma=0.1)
    """The speed distribution of this actor in m/s."""
    imperfection: Distribution = Distribution(mean=0.5, sigma=0)
    """Driver imperfection within range [0..1]"""
    min_gap: Distribution = Distribution(mean=2.5, sigma=0)
    """Minimum gap (when standing) in meters."""
    max_speed: float = 55.5
    """The vehicle's maximum velocity (in m/s), defaults to 200 km/h for vehicles"""
    vehicle_type: str = "passenger"
    """The configured vehicle type this actor will perform as. ("passenger", "bus", "coach", "truck", "trailer")"""
    lane_changing_model: LaneChangingModel = field(
        default_factory=LaneChangingModel, hash=False
    )
    junction_model: JunctionModel = field(default_factory=JunctionModel, hash=False)

    def __hash__(self) -> int:
        return pickle_hash_int(self)

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
    of 'path.to.file:locator-name' where the path to the registration call is in the form
    {PYTHONPATH}[n]/path/to/file.py
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
    capacity: "BubbleLimits" = None
    """The capacity of the boid agent to take over vehicles."""


# A MapBuilder should return an object derived from the RoadMap base class
# and a hash that uniquely identifies it (changes to the hash should signify
# that the map is different enough that map-related caches should be reloaded).
#
# This function should be re-callable (although caching is up to the implementation).
# The idea here is that anything in SMARTS that needs to use a RoadMap
# can call this builder to get or create one as necessary.
MapBuilder = Callable[[Any], Tuple[Optional[RoadMap], Optional[str]]]


@dataclass(frozen=True)
class MapSpec:
    """A map specification that describes how to generate a roadmap."""

    source: str
    """A path or URL or name uniquely designating the map source."""
    lanepoint_spacing: Optional[float] = None
    """If specified, the default distance between pre-generated Lane Points (Waypoints)."""
    default_lane_width: Optional[float] = None
    """If specified, the default width (in meters) of lanes on this map."""
    shift_to_origin: bool = False
    """If True, upon creation a map whose bounding-box does not intersect with
    the origin point (0,0) will be shifted such that it does."""
    builder_fn: MapBuilder = get_road_map
    """If specified, this should return an object derived from the RoadMap base class
    and a hash that uniquely identifies it (changes to the hash should signify
    that the map is different enough that map-related caches should be reloaded).
    The parameter is this MapSpec object itself.
    If not specified, this currently defaults to a function that creates
    SUMO road networks (get_road_map()) in smarts.core.default_map_builder."""


@dataclass(frozen=True)
class Route:
    """A route is represented by begin and end road IDs, with an optional list of
    intermediary road IDs. When an intermediary is not specified the router will
    decide what it should be.
    """

    ## road, lane index, offset
    begin: Tuple[str, int, Any]
    """The (road, lane_index, offset) details of the start location for the route.

    road:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: "max", "random"
    """
    ## road, lane index, offset
    end: Tuple[str, int, Any]
    """The (road, lane_index, offset) details of the end location for the route.

    road:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: "max", "random"
    """

    # Roads we want to make sure this route includes
    via: Tuple[str, ...] = field(default_factory=tuple)
    """The ids of roads that must be included in the route between `begin` and `end`."""

    map_spec: Optional[MapSpec] = None
    """All routes are relative to a road map.  If not specified here,
    the default map_spec for the scenario is used."""

    @property
    def id(self) -> str:
        """The unique id of this route."""
        return "route-{}-{}-{}-".format(
            "_".join(map(str, self.begin)),
            "_".join(map(str, self.end)),
            pickle_hash_int(self),
        )

    @property
    def roads(self):
        """All roads that are used within this route."""
        return (self.begin[0],) + self.via + (self.end[0],)

    def __hash__(self):
        return pickle_hash_int(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


@dataclass(frozen=True)
class RandomRoute:
    """An alternative to types.Route which specifies to sstudio to generate a random
    route.
    """

    id: str = field(default_factory=lambda: f"random-route-{gen_id()}")

    map_spec: Optional[MapSpec] = None
    """All routes are relative to a road map.  If not specified here,
    the default map_spec for the scenario is used."""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and hash(self) == hash(other)


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
    """An actor to weight mapping associated as\\: { actor: weight }

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
        return "flow-{}-{}-".format(
            self.route.id,
            str(pickle_hash_int(sorted(self.actors.items(), key=lambda a: a[0].name))),
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
    actor: TrafficActor = field(init=False)
    """The traffic actor model (usually vehicle) that will be used for the trip."""

    def __post_init__(self):
        object.__setattr__(
            self,
            "actor",
            TrafficActor(name=self.vehicle_name, vehicle_type=self.vehicle_type),
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
class JunctionEdgeIDResolver:
    """A utility for resolving a junction connection edge"""

    start_edge_id: str
    start_lane_index: int
    end_edge_id: str
    end_lane_index: int

    def to_edge(self, sumo_road_network) -> str:
        """Queries the road network to see if there is a junction edge between the two
        given edges.
        """
        return sumo_road_network.get_edge_in_junction(
            self.start_edge_id,
            self.start_lane_index,
            self.end_edge_id,
            self.end_lane_index,
        )


@dataclass
class Via:
    """A point on a road that an actor must pass through"""

    road_id: Union[str, JunctionEdgeIDResolver]
    """The road this via is on"""
    lane_index: int
    """The lane this via sits on"""
    lane_offset: int
    """The offset along the lane where this via sits"""
    required_speed: float
    """The speed that a vehicle should travel through this via"""
    hit_distance: float = -1
    """The distance at which this waypoint can be hit. Negative means half the lane radius."""


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


@dataclass(frozen=True)
class EntryTactic:
    """The tactic that the simulation should use to acquire a vehicle for an actor."""

    pass


@dataclass(frozen=True)
class TrapEntryTactic(EntryTactic):
    """An entry tactic that repurposes a pre-existing vehicle for an actor."""

    wait_to_hijack_limit_s: float
    """The amount of seconds a hijack will wait to get a vehicle before defaulting to a new vehicle"""
    zone: Optional["MapZone"] = None
    """The zone of the hijack area"""
    exclusion_prefixes: Tuple[str, ...] = tuple()
    """The prefixes of vehicles to avoid hijacking"""
    default_entry_speed: Optional[float] = None
    """The speed that the vehicle starts at when the hijack limit expiry emits a new vehicle"""


@dataclass(frozen=True)
class Mission:
    """The descriptor for an actor's mission."""

    route: Union[RandomRoute, Route]
    """The route for the actor to attempt to follow."""

    via: Tuple[Via, ...] = ()
    """Points on an road that an actor must pass through"""

    start_time: float = 0.1
    """The earliest simulation time that this mission starts but may start later in couple with
    `entry_tactic`.
    """

    entry_tactic: Optional[EntryTactic] = None
    """A specific tactic the mission should employ to start the mission."""


@dataclass(frozen=True)
class EndlessMission:
    """The descriptor for an actor's mission that has no end."""

    begin: Tuple[str, int, float]
    """The (road, lane_index, offset) details of the start location for the route.

    road:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: 'max', 'random'
    """
    via: Tuple[Via, ...] = ()
    """Points on a road that an actor must pass through"""
    start_time: float = 0.1
    """The earliest simulation time that this mission starts"""
    entry_tactic: Optional[EntryTactic] = None
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
    via: Tuple[Via, ...] = ()
    """Points on a road that an actor must pass through"""
    start_time: float = 0.1
    """The earliest simulation time that this mission starts"""
    entry_tactic: Optional[EntryTactic] = None
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
    via: Tuple[Via, ...] = ()
    """Points on a road that an actor must pass through"""


@dataclass(frozen=True)
class Zone:
    """The base for a descriptor that defines a capture area."""

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generates the geometry from this zone."""
        raise NotImplementedError


@dataclass(frozen=True)
class MapZone(Zone):
    """A descriptor that defines a capture area."""

    start: Tuple[str, int, float]
    """The (road_id, lane_index, offset) details of the starting location.

    road_id:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in metres into the lane. Also acceptable\\: 'max', 'random'
    """
    length: float
    """The length of the geometry along the center of the lane. Also acceptable\\: 'max'"""
    n_lanes: int = 2
    """The number of lanes from right to left that this zone covers."""

    def to_geometry(self, road_map: RoadMap) -> Polygon:
        """Generates a map zone over a stretch of the given lanes."""

        def resolve_offset(offset, geometry_length, lane_length):
            if offset == "base":
                return 0
            # push off of end of lane
            elif offset == "max":
                return lane_length - geometry_length
            elif offset == "random":
                return random.uniform(0, lane_length - geometry_length)
            else:
                return float(offset)

        def pick_remaining_shape_after_split(geometry_collection, expected_point):
            lane_shape = geometry_collection
            if not isinstance(lane_shape, GeometryCollection):
                return lane_shape

            # For simplicity, we only deal w/ the == 1 or 2 case
            if len(lane_shape.geoms) not in {1, 2}:
                return None

            if len(lane_shape.geoms) == 1:
                return lane_shape.geoms[0]

            # We assume that there are only two split shapes to choose from
            keep_index = 0
            if lane_shape.geoms[1].minimum_rotated_rectangle.contains(expected_point):
                # 0 is the discard piece, keep the other
                keep_index = 1

            lane_shape = lane_shape.geoms[keep_index]

            return lane_shape

        def split_lane_shape_at_offset(
            lane_shape: Polygon, lane: RoadMap.Lane, offset: float
        ):
            # XXX: generalize to n-dim
            width_2, _ = lane.width_at_offset(offset)
            point = np.array(lane.from_lane_coord(RefLinePoint(offset)))[:2]
            lane_vec = lane.vector_at_offset(offset)[:2]

            perp_vec_right = rotate_cw_around_point(lane_vec, np.pi / 2, origin=(0, 0))
            perp_vec_right = (
                perp_vec_right / max(np.linalg.norm(perp_vec_right), 1e-3) * width_2
                + point
            )

            perp_vec_left = rotate_cw_around_point(lane_vec, -np.pi / 2, origin=(0, 0))
            perp_vec_left = (
                perp_vec_left / max(np.linalg.norm(perp_vec_left), 1e-3) * width_2
                + point
            )

            split_line = LineString([perp_vec_left, perp_vec_right])
            return split(lane_shape, split_line)

        lane_shapes = []
        road_id, lane_idx, offset = self.start
        road = road_map.road_by_id(road_id)
        buffer_from_ends = 1e-6
        for lane_idx in range(lane_idx, lane_idx + self.n_lanes):
            lane = road.lane_at_index(lane_idx)
            lane_length = lane.length
            geom_length = self.length

            if geom_length > lane_length:
                logging.debug(
                    f"Geometry is too long={geom_length} with offset={offset} for "
                    f"lane={lane.lane_id}, using length={lane_length} instead"
                )
                geom_length = lane_length

            assert geom_length > 0  # Geom length is negative

            lane_offset = resolve_offset(offset, geom_length, lane_length)
            lane_offset += buffer_from_ends
            width, _ = lane.width_at_offset(lane_offset)  # TODO
            lane_shape = lane.shape(0.3, width)  # TODO

            geom_length = max(geom_length - buffer_from_ends, buffer_from_ends)
            lane_length = max(lane_length - buffer_from_ends, buffer_from_ends)

            min_cut = min(lane_offset, lane_length)
            # Second cut takes into account shortening of geometry by `min_cut`.
            max_cut = min(min_cut + geom_length, lane_length)

            midpoint = Point(
                *lane.from_lane_coord(RefLinePoint(s=lane_offset + geom_length * 0.5))
            )

            lane_shape = split_lane_shape_at_offset(lane_shape, lane, min_cut)
            lane_shape = pick_remaining_shape_after_split(lane_shape, midpoint)
            if lane_shape is None:
                continue

            lane_shape = split_lane_shape_at_offset(
                lane_shape,
                lane,
                max_cut,
            )
            lane_shape = pick_remaining_shape_after_split(lane_shape, midpoint)
            if lane_shape is None:
                continue

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
    rotation: Optional[float] = None
    """The heading direction of the bubble. (radians, clock-wise rotation)"""

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generates a box zone at the given position."""
        w, h = self.size
        x, y = self.pos[:2]
        p0 = (-w / 2, -h / 2)  # min
        p1 = (w / 2, h / 2)  # max
        poly = Polygon([p0, (p0[0], p1[1]), p1, (p1[0], p0[1])])
        if self.rotation is not None:
            poly = shapely_rotate(poly, self.rotation, use_radians=True)
        return shapely_translate(poly, xoff=x, yoff=y)


@dataclass(frozen=True)
class ConfigurableZone(Zone):
    """A descriptor for a zone with user-defined geometry."""

    ext_coordinates: List[Tuple[float, float]]
    """external coordinates of the polygon
    < 2 points provided: error
    = 2 points provided: generates a box using these two points as diagonal
    > 2 points provided: generates a polygon according to the coordinates"""
    rotation: Optional[float] = None
    """The heading direction of the bubble(radians, clock-wise rotation)"""

    def __post_init__(self):
        if (
            not self.ext_coordinates
            or len(self.ext_coordinates) < 2
            or not isinstance(self.ext_coordinates[0], tuple)
        ):
            raise ValueError(
                "Two points or more are needed to create a polygon. (less than two points are provided)"
            )

        x_set = set(point[0] for point in self.ext_coordinates)
        y_set = set(point[1] for point in self.ext_coordinates)
        if len(x_set) == 1 or len(y_set) == 1:
            raise ValueError(
                "Parallel line cannot form a polygon. (points provided form a parallel line)"
            )

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generate a polygon according to given coordinates"""
        poly = None
        if (
            len(self.ext_coordinates) == 2
        ):  # if user only specified two points, create a box
            x_min = min(self.ext_coordinates[0][0], self.ext_coordinates[1][0])
            x_max = max(self.ext_coordinates[0][0], self.ext_coordinates[1][0])
            y_min = min(self.ext_coordinates[0][1], self.ext_coordinates[1][1])
            y_max = max(self.ext_coordinates[0][1], self.ext_coordinates[1][1])
            poly = box(x_min, y_min, x_max, y_max)

        else:  # else create a polygon according to the coordinates
            poly = Polygon(self.ext_coordinates)

        if self.rotation is not None:
            poly = shapely_rotate(poly, self.rotation, use_radians=True)
        return poly


@dataclass(frozen=True)
class BubbleLimits:
    """Defines the capture limits of a bubble."""

    hijack_limit: int = maxsize
    """The maximum number of vehicles the bubble can hijack"""
    shadow_limit: int = maxsize
    """The maximum number of vehicles the bubble can shadow"""

    def __post_init__(self):
        if self.shadow_limit is None:
            raise ValueError("Shadow limit must be a non-negative real number")
        if self.hijack_limit is None or self.shadow_limit < self.hijack_limit:
            raise ValueError("Shadow limit must be >= hijack limit")


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


@dataclass(frozen=True)
class RoadSurfacePatch:
    """A descriptor that defines a patch of road surface with a different friction coefficient."""

    zone: Zone
    """The zone which to capture vehicles."""
    begin_time: int
    """The start time in seconds of when this surface is active."""
    end_time: int
    """The end time in seconds of when this surface is active."""
    friction_coefficient: float
    """The surface friction coefficient."""


@dataclass(frozen=True)
class ActorAndMission:
    """Holds an Actor object and its associated Mission."""

    actor: Actor
    """Specification for traffic actor.
    """
    mission: Union[Mission, EndlessMission, LapMission]
    """Mission for traffic actor.
    """


@dataclass(frozen=True)
class TrafficHistoryDataset:
    """Describes a dataset containing trajectories (time-stamped positions)
    for a set of vehicles.  Often these have been collected by third parties
    from real-world observations, hence the name 'history'.  When used
    with a SMARTS scenario, traffic vehicles will move on the map according
    to their trajectories as specified in the dataset.  These can be mixed
    with other types of traffic (such as would be specified by an object of
    the Traffic type in this DSL).  In order to use this efficiently, SMARTS
    will pre-process ('import') the dataset when the scenario is built."""

    name: str
    """a unique name for the dataset"""
    source_type: str
    """the type of the dataset; supported values include: NGSIM, INTERACTION, Waymo"""
    input_path: Optional[str] = None
    """a relative or absolute path to the dataset; if omitted, dataset will not be imported"""
    scenario_id: Optional[str] = None
    """a unique ID for a Waymo scenario. For other datasets, this field will be None."""
    x_margin_px: float = 0.0
    """x offset of the map from the data (in pixels)"""
    y_margin_px: float = 0.0
    """y offset of the map from the data (in pixels)"""
    swap_xy: bool = False
    """if True, the x and y axes the dataset coordinate system will be swapped"""
    flip_y: bool = False
    """if True, the dataset will be mirrored around the x-axis"""
    filter_off_map: bool = False
    """if True, then any vehicle whose coordinates on a time step fall outside of the map's bounding box will be removed for that time step"""

    map_lane_width: float = 3.7
    """This is used to figure out the map scale, which is map_lane_width / real_lane_width_m.  (So use `real_lane_width_m` here for 1:1 scale - the default.)  It's also used in SMARTS for detecting off-road, etc."""
    real_lane_width_m: float = 3.7
    """Average width in meters of the dataset's lanes in the real world.  US highway lanes are about 12 feet (or ~3.7m, the default) wide."""
    speed_limit_mps: Optional[float] = None
    """used by SMARTS for the initial speed of new agents being added to the scenario"""

    heading_inference_window: int = 2
    """When inferring headings from positions, a sliding window (moving average) of this size will be used to smooth inferred headings and reduce their dependency on any individual position changes.  Defaults to 2 if not specified."""
    heading_inference_min_speed: float = 2.2
    """Speed threshold below which a vehicle's heading is assumed not to change.  This is useful to prevent abnormal heading changes that may arise from noise in position estimates in a trajectory dataset dominating real position changes in situations where the real position changes are very small.  Defaults to 2.2 m/s if not specified."""
    max_angular_velocity: Optional[float] = None
    """When inferring headings from positions, each vehicle's angular velocity will be limited to be at most this amount (in rad/sec) to prevent lateral-coordinate noise in the dataset from causing near-instantaneous heading changes."""
    default_heading: float = 1.5 * math.pi
    """A heading in radians to be used by default for vehicles if the headings are not present in the dataset and cannot be inferred from position changes (such as on the first time step)."""


@dataclass(frozen=True)
class Scenario:
    """The sstudio scenario representation."""

    map_spec: Optional[MapSpec] = None
    """Specifies the road map."""
    traffic: Optional[Dict[str, Traffic]] = None
    """Background traffic vehicle specification."""
    ego_missions: Optional[Sequence[Mission]] = None
    """Ego agent missions."""
    social_agent_missions: Optional[
        Dict[str, Tuple[Sequence[SocialAgentActor], Sequence[Mission]]]
    ] = None
    """
    Every dictionary item ``{group: (actors, missions)}`` gets run simultaneously.
    If actors > 1 and missions = 0 or actors = 1 and missions > 0, we cycle
    through them every episode. Otherwise actors must be the same length as 
    missions.
    """
    bubbles: Optional[Sequence[Bubble]] = None
    """Capture bubbles for focused social agent simulation."""
    friction_maps: Optional[Sequence[RoadSurfacePatch]] = None
    """Friction coefficient of patches of road surface."""
    traffic_histories: Optional[Sequence[Union[TrafficHistoryDataset, str]]] = None
    """Traffic vehicles trajectory dataset to be replayed."""
