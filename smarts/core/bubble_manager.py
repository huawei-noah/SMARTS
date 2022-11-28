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
import logging
import math
from builtins import classmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from sys import maxsize
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from shapely.affinity import rotate, translate
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Point, Polygon

from smarts.core.data_model import SocialAgent
from smarts.core.plan import (
    EndlessGoal,
    Mission,
    Plan,
    PlanningError,
    PositionalGoal,
    Start,
)
from smarts.core.road_map import RoadMap
from smarts.core.utils.id import SocialAgentId
from smarts.core.utils.string import truncate
from smarts.core.vehicle import Vehicle
from smarts.core.vehicle_index import VehicleIndex
from smarts.sstudio.types import BoidAgentActor
from smarts.sstudio.types import Bubble as SSBubble
from smarts.sstudio.types import BubbleLimits, SocialAgentActor
from smarts.zoo.registry import make as make_social_agent


class BubbleTransition(Enum):
    """Describes a bubble transition state."""

    # --> [  AirlockEntered  [  Entered  ]  Exited  ]  AirlockExited -->
    AirlockEntered = 0  # --> (Airlock)
    Entered = 1  # Airlock --> (Bubble)
    Exited = 2  # Bubble --> (Airlock)
    AirlockExited = 3  # Airlock -->


class BubbleState(Enum):
    """Describes the actor's occupancy in a bubble."""

    InBubble = 0
    InAirlock = 1
    WasInBubble = 2


class Bubble:
    """Wrapper around `sstudio.types.Bubble` to cache bubble geometry (and does some
    internal Bubble-related business logic).
    """

    def __init__(self, bubble: SSBubble, road_map: RoadMap):
        geometry = bubble.zone.to_geometry(road_map)

        self.centroid = [geometry.centroid.x, geometry.centroid.y]
        bounding = geometry.bounds
        self.radius = math.sqrt(
            pow(bounding[2] - self.centroid[0], 2)
            + pow(bounding[1] - self.centroid[1], 2)
        )

        bubble_limit = (
            bubble.limit or BubbleLimits()
            if bubble.limit is None or isinstance(bubble.limit, BubbleLimits)
            else BubbleLimits(bubble.limit.hijack_limit, bubble.limit.shadow_limit + 1)
        )

        if isinstance(bubble.actor, BoidAgentActor):

            def safe_min(a, b):
                return min(a or maxsize, b or maxsize)

            if bubble.limit is None:
                bubble_limit = bubble.actor.capacity
            elif bubble.actor.capacity is not None:
                hijack_limit = safe_min(
                    bubble.limit.hijack_limit, bubble.actor.capacity
                )
                shadow_limit = safe_min(
                    bubble.limit.shadow_limit, bubble.actor.capacity.shadow_limit
                )
                bubble_limit = BubbleLimits(hijack_limit, shadow_limit)

        self._bubble_heading = 0.0
        self._bubble = bubble
        self._limit = bubble_limit
        self._cached_inner_geometry = geometry
        self._exclusion_prefixes = bubble.exclusion_prefixes

        self._cached_airlock_geometry = self._cached_inner_geometry.buffer(
            bubble.margin,
            cap_style=CAP_STYLE.square,
            join_style=JOIN_STYLE.mitre,
        )

    @property
    def exclusion_prefixes(self):
        """The blacklist of actor prefixes, used to ignore specific actors."""
        return self._exclusion_prefixes

    @property
    def id(self):
        """The id of the underlying bubble."""
        return self._bubble.id

    @property
    def actor(self) -> SocialAgentActor:
        """The actor that should replace the captured actor."""
        return self._bubble.actor

    @property
    def follow_actor_id(self) -> str:
        """A target actor that the bubble should remain at a fixed offset from."""
        return self._bubble.follow_actor_id

    @property
    def follow_vehicle_id(self) -> str:
        """A target vehicle that the bubble should remain at a fixed offset from."""
        return self._bubble.follow_vehicle_id

    @property
    def limit(self):
        """The maximum number of actors that the bubble can have captured at the same time."""
        return self._limit

    @property
    def is_boid(self):
        """If the actors captured by the bubble should be controlled by a boid agent."""
        return self._bubble.is_boid

    @property
    def keep_alive(self):
        """If enabled, the social agent actor will be spawned upon first vehicle airlock
        and be reused for every subsequent vehicle entering the bubble until the episode
        is over.
        """
        return self._bubble.keep_alive

    # XXX: In the case of travelling bubbles, the geometry and zone are moving
    #      according to the follow vehicle.
    @property
    def geometry(self) -> Polygon:
        """The geometry of the managed bubble."""
        return self._cached_inner_geometry

    @property
    def airlock_geometry(self) -> Polygon:
        """The airlock geometry of the managed bubble."""
        return self._cached_airlock_geometry

    def admissibility(
        self,
        vehicle_id: str,
        index: VehicleIndex,
        vehicle_ids_in_bubbles: Dict["Bubble", Set[str]],
        running_cursors: Set["Cursor"],
    ):
        """The vehicle_id we are querying for and the `other_vehicle_ids` _presently in
        this `sstudio.types.Bubble`.
        """
        for prefix in self.exclusion_prefixes:
            if vehicle_id.startswith(prefix):
                return False, False

        hijackable, shadowable = True, True
        if self._limit is not None:
            # Already hijacked (according to VehicleIndex) + to be hijacked (running cursors)
            current_hijacked_vehicle_ids = {
                v_id
                for v_id in vehicle_ids_in_bubbles[self]
                if index.vehicle_is_hijacked(v_id)
            }
            current_shadowed_vehicle_ids = {
                v_id
                for v_id in vehicle_ids_in_bubbles[self]
                if index.vehicle_is_shadowed(v_id)
            }
            vehicle_ids_by_bubble_state = (
                BubbleManager._vehicle_ids_divided_by_bubble_state(
                    frozenset(running_cursors)
                )
            )

            # pytype: disable=unsupported-operands
            all_hijacked_vehicle_ids = (
                current_hijacked_vehicle_ids
                | vehicle_ids_by_bubble_state[BubbleState.InAirlock][self]
            ) - {vehicle_id}

            all_shadowed_vehicle_ids = (
                current_shadowed_vehicle_ids
                | vehicle_ids_by_bubble_state[BubbleState.InBubble][self]
            ) - {vehicle_id}
            # pytype: enable=unsupported-operands

            hijackable = len(all_hijacked_vehicle_ids) < (
                self._limit.hijack_limit or maxsize
            )
            shadowable = len(all_shadowed_vehicle_ids) + len(
                all_hijacked_vehicle_ids
            ) < (self._limit.shadow_limit or maxsize)

        return hijackable, shadowable

    def in_bubble_or_airlock(self, position: Point):
        """Test if the position is within the bubble or airlock around the bubble."""
        if not isinstance(position, Point):
            position = Point(position)

        in_airlock = position.within(self._cached_airlock_geometry)
        if not in_airlock:
            return False, False

        in_bubble = position.within(self._cached_inner_geometry)
        return in_bubble, in_airlock and not in_bubble

    @property
    def is_travelling(self):
        """If the bubble is following an actor."""
        return (
            self._bubble.follow_actor_id is not None
            or self._bubble.follow_vehicle_id is not None
        )

    def move_to_follow_vehicle(self, vehicle: Vehicle):
        """Move the bubble to a pose relative to the given vehicle."""
        if not vehicle.valid:
            return

        x, y, _ = vehicle.position

        def _transform(geom):
            centroid = geom.centroid

            # Bring back to origin
            geom = translate(geom, xoff=-centroid.x, yoff=-centroid.y)
            geom = rotate(geom, -self._bubble_heading, "centroid", use_radians=True)

            # Now apply new transformation in "vehicle coordinate space"
            geom = translate(
                geom,
                xoff=self._bubble.follow_offset[0],
                yoff=self._bubble.follow_offset[1],
            )
            geom = rotate(geom, vehicle.heading, (0, 0), use_radians=True)
            geom = translate(geom, xoff=x, yoff=y)
            return geom

        self._cached_inner_geometry = _transform(self._cached_inner_geometry)
        self._cached_airlock_geometry = _transform(self._cached_airlock_geometry)
        self.centroid = [
            self._cached_inner_geometry.centroid.x,
            self._cached_inner_geometry.centroid.y,
        ]

        self._bubble_heading = vehicle.heading

    def __repr__(self):
        return f"""Bubble(
  id={self.id},
  travelling={self.is_travelling},
  actor={self.actor},
  follow_actor_id={self.follow_actor_id},
  limit={self.limit},
  geometry={self.geometry},
  airlock_geometry={self.airlock_geometry},
  follow_vehicle_id={self.follow_vehicle_id},
)"""

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True, eq=True)
class Cursor:
    """Tracks an actor through an airlock or a bubble."""

    # We would always want to have the vehicle go through the airlock zone. This may
    # not be the case if we spawn a vehicle in a bubble, but that wouldn't be ideal.
    vehicle_id: str
    state: Optional[BubbleState] = None
    transition: Optional[BubbleTransition] = None
    bubble: Optional[Bubble] = None

    @classmethod
    def for_removed(
        cls,
        vehicle_id: str,
        bubble: Bubble,
        index: VehicleIndex,
        vehicle_ids_per_bubble: Dict[Bubble, Set[str]],
    ) -> "Cursor":
        """Generate a cursor for an inactive bubble.
        Args:
            vehicle (Vehicle):
                The vehicle that is to be tracked.
            bubble (Bubble):
                The bubble that the vehicle is interacting with.
            index (VehicleIndex):
                The vehicle index the vehicle is in.
            vehicle_ids_per_bubble (Dict[Bubble, Set[str]]):
                Bubbles associated with vehicle ids.
            running_cursors (Set["Cursor"]):
                A set of existing cursors.
        """
        was_in_this_bubble = vehicle_id in vehicle_ids_per_bubble[bubble]
        is_hijacked, is_shadowed = index.vehicle_is_hijacked_or_shadowed(vehicle_id)
        transition = None
        if was_in_this_bubble and (is_shadowed or is_hijacked):
            transition = BubbleTransition.AirlockExited
        return cls(
            vehicle_id=vehicle_id,
            transition=transition,
            state=BubbleState.WasInBubble,
            bubble=bubble,
        )

    @classmethod
    def from_pos(
        cls,
        pos: Point,
        vehicle_id: str,
        bubble: Bubble,
        index: VehicleIndex,
        vehicle_ids_per_bubble: Dict[Bubble, Set[str]],
        running_cursors: Set["Cursor"],
    ) -> "Cursor":
        """Generate a cursor.
        Args:
            pos (Point):
                The shapely position of the vehicle.
            vehicle (Vehicle):
                The vehicle that is to be tracked.
            bubble (Bubble):
                The bubble that the vehicle is interacting with.
            index (VehicleIndex):
                The vehicle index the vehicle is in.
            vehicle_ids_per_bubble (Dict[Bubble, Set[str]]):
                Bubbles associated with vehicle ids.
            running_cursors (Set["Cursor"]):
                A set of existing cursors.
        """
        in_bubble_zone, in_airlock_zone = bubble.in_bubble_or_airlock(pos)
        is_social = vehicle_id in index.social_vehicle_ids()
        is_hijacked, is_shadowed = index.vehicle_is_hijacked_or_shadowed(vehicle_id)
        is_hijack_admissible, is_airlock_admissible = bubble.admissibility(
            vehicle_id, index, vehicle_ids_per_bubble, running_cursors
        )
        was_in_this_bubble = vehicle_id in vehicle_ids_per_bubble[bubble]

        # XXX: When a travelling bubble disappears and an agent is airlocked or
        #      hijacked. It remains in that state.
        # TODO: Depending on step size, we could potentially skip transitions (e.g.
        #       go straight to relinquish w/o hijacking first). This may be solved by
        #       time-based airlocking. For robust code we'll want to handle these
        #       scenarios (e.g. hijacking if didn't airlock first)
        transition = None
        if (
            is_social
            and not is_shadowed
            and is_airlock_admissible
            and (in_airlock_zone or in_bubble_zone)
        ):
            # In this case a vehicle has just entered the airlock
            transition = BubbleTransition.AirlockEntered
        elif is_social and is_shadowed and is_hijack_admissible and in_bubble_zone:
            # In this case a vehicle has just entered the bubble
            transition = BubbleTransition.Entered
        elif was_in_this_bubble and is_hijacked and in_airlock_zone:
            # XXX: This may get called repeatedly because we don't actually change
            #      any state when this happens.
            # In this case a vehicle has just exited the bubble
            transition = BubbleTransition.Exited
        elif (
            was_in_this_bubble
            and (is_shadowed or is_hijacked)
            and not (in_airlock_zone or in_bubble_zone)
        ):
            # In this case a vehicle has just exited the airlock around the bubble
            transition = BubbleTransition.AirlockExited

        state = None
        if in_bubble_zone:
            state = BubbleState.InBubble
        elif in_airlock_zone:
            state = BubbleState.InAirlock

        return cls(
            vehicle_id=vehicle_id, transition=transition, state=state, bubble=bubble
        )

    def __repr__(self):
        return f"Cursor(state={self.state}, transition={self.transition}, vehicle_id={self.vehicle_id})"

    def __hash__(self) -> int:
        return hash((self.vehicle_id, self.state, self.transition, self.bubble.id))


class BubbleManager:
    """Manages bubble interactions."""

    def __init__(self, bubbles: Sequence[SSBubble], road_map: RoadMap):
        self._log = logging.getLogger(self.__class__.__name__)
        self._cursors: Set[Cursor] = set()
        self._last_vehicle_index = VehicleIndex.identity()
        self._bubbles = [Bubble(b, road_map) for b in bubbles]

    @property
    def bubbles(self) -> Sequence[Bubble]:
        """A sequence of currently active bubbles."""
        active_bubbles, _ = self._bubble_groups()
        return active_bubbles

    def _bubble_groups(self) -> Tuple[List[Bubble], List[Bubble]]:
        # Filter out travelling bubbles that are missing their follow vehicle
        def is_active(bubble):
            if not bubble.is_travelling:
                return True

            vehicles = []
            if bubble.follow_actor_id is not None:
                vehicles += self._last_vehicle_index.vehicles_by_actor_id(
                    bubble.follow_actor_id
                )
            if bubble.follow_vehicle_id is not None:
                vehicle = self._last_vehicle_index.vehicle_by_id(
                    bubble.follow_vehicle_id, None
                )
                if vehicle is not None:
                    vehicles += [vehicle]
            if len(vehicles) > 1:
                logging.error(
                    f"bubble `{bubble.id} follows multiple vehicles: {[v.id for v in vehicles]}"
                )
            return len(vehicles) == 1

        active_bubbles = []
        inactive_bubbles = []
        for bubble in self._bubbles:
            if is_active(bubble):
                active_bubbles.append(bubble)
            else:
                inactive_bubbles.append(bubble)
        return active_bubbles, inactive_bubbles

    @staticmethod
    @lru_cache(maxsize=2)
    def _vehicle_ids_divided_by_bubble_state(
        cursors: FrozenSet[Cursor],
    ) -> Dict[Bubble, Set[Bubble]]:
        vehicle_ids_grouped_by_cursor = defaultdict(lambda: defaultdict(set))
        for cursor in cursors:
            vehicle_ids_grouped_by_cursor[cursor.state][cursor.bubble].add(
                cursor.vehicle_id
            )
        return vehicle_ids_grouped_by_cursor

    def vehicle_ids_per_bubble(
        self,
    ) -> Dict[Bubble, Set[str]]:
        """Bubbles associated with the vehicles they contain."""
        vid = self._vehicle_ids_divided_by_bubble_state(frozenset(self._cursors))
        return defaultdict(
            set, {**vid[BubbleState.InBubble], **vid[BubbleState.InAirlock]}
        )

    def agent_ids_for_bubble(self, bubble: Bubble, sim) -> Set[str]:
        """Agents generated by this bubble."""
        bubble_cursors = set(filter(lambda c: c.bubble == bubble, self._cursors))

        agent_ids = set()
        for bc in bubble_cursors:
            if bc.state != BubbleState.InBubble:
                continue
            agent_id = sim.vehicle_index.actor_id_from_vehicle_id(bc.vehicle_id)
            if agent_id is not None:
                agent_ids.add(agent_id)
        return agent_ids

    def step(self, sim):
        """Update the associations between bubbles, actors, and agents"""
        self._move_travelling_bubbles(sim)
        self._cursors = self._sync_cursors(self._last_vehicle_index, sim.vehicle_index)
        self._handle_transitions(sim, self._cursors)
        self._last_vehicle_index = deepcopy(sim.vehicle_index)

    def _sync_cursors(self, last_vehicle_index, vehicle_index):
        # TODO: Not handling newly added vehicles means we require an additional step
        #       before we trigger hijacking.
        # Newly added vehicles
        # add_index = vehicle_index - last_vehicle_index
        # TODO: Not handling deleted vehicles at this point should be fine because we're
        #       stateless.
        # Recently terminated vehicles
        # del_index = last_vehicle_index - vehicle_index

        # Vehicles that stuck around
        persisted_vehicle_index = vehicle_index & last_vehicle_index

        # Calculate latest cursors
        vehicle_ids_per_bubble = self.vehicle_ids_per_bubble()
        cursors = set()
        active_bubbles, inactive_bubbles = self._bubble_groups()
        inactive_bubbles_to_run = [
            b for b in inactive_bubbles if len(vehicle_ids_per_bubble[b])
        ]

        if inactive_bubbles_to_run:
            for bubble in inactive_bubbles_to_run:
                for _, vehicle in persisted_vehicle_index.vehicleitems():
                    cursor = Cursor.for_removed(
                        vehicle_id=vehicle.id,
                        bubble=bubble,
                        index=persisted_vehicle_index,
                        vehicle_ids_per_bubble=vehicle_ids_per_bubble,
                    )
                    if cursor.transition not in (BubbleTransition.AirlockExited,):
                        continue
                    cursors.add(cursor)

        if not active_bubbles:
            return cursors

        for _, vehicle in persisted_vehicle_index.vehicleitems():
            # XXX: Turns out Shapely Point(...) creation is very expensive (~0.02ms) which
            #      when inside of a loop x large number of vehicles makes a big
            #      performance hit.
            point = vehicle.pose.point
            v_radius = math.sqrt(
                vehicle.width * vehicle.width + vehicle.length * vehicle.length
            )

            for bubble in active_bubbles:
                was_in_this_bubble = vehicle_ids_per_bubble[bubble]
                sq_distance = (point.x - bubble.centroid[0]) * (
                    point.x - bubble.centroid[0]
                ) + (point.y - bubble.centroid[1]) * (point.y - bubble.centroid[1])

                if vehicle.id in was_in_this_bubble or sq_distance <= pow(
                    v_radius + bubble.radius + bubble._bubble.margin, 2
                ):
                    cursor = Cursor.from_pos(
                        pos=point.as_shapely,
                        vehicle_id=vehicle.id,
                        bubble=bubble,
                        index=persisted_vehicle_index,
                        vehicle_ids_per_bubble=vehicle_ids_per_bubble,
                        running_cursors=cursors,
                    )
                    cursors.add(cursor)

        return cursors

    def _handle_transitions(self, sim, cursors: Set[Cursor]):
        social_agent_vehicles = []
        for agent_id in sim.agent_manager.social_agent_ids:
            social_agent_vehicles += sim.vehicle_index.vehicles_by_actor_id(agent_id)

        transitioned = [c for c in cursors if c.transition is not None]
        for cursor in transitioned:
            if cursor.transition == BubbleTransition.AirlockEntered:
                self._airlock_social_vehicle_with_social_agent(
                    sim, cursor.vehicle_id, cursor.bubble.actor, cursor.bubble
                )
            elif cursor.transition == BubbleTransition.Entered:
                self._hijack_social_vehicle_with_social_agent(
                    sim, cursor.vehicle_id, cursor.bubble.actor, cursor.bubble
                )
            elif cursor.transition == BubbleTransition.Exited:
                continue
            elif cursor.transition == BubbleTransition.AirlockExited:
                teardown = not cursor.bubble.is_boid or not cursor.bubble.keep_alive
                sim.vehicle_exited_bubble(cursor.vehicle_id, teardown)

    def _move_travelling_bubbles(self, sim):
        active_bubbles, inactive_bubbles = self._bubble_groups()
        for bubble in [*active_bubbles, *inactive_bubbles]:
            if not bubble.is_travelling:
                continue

            vehicles = []
            # XXX: Using a vehicle reference through the `_last_vehicle_index` is a
            # XXX clear error since it can reference out of date vehicles.
            if bubble.follow_actor_id is not None:
                vehicles += self._last_vehicle_index.vehicles_by_actor_id(
                    bubble.follow_actor_id
                )
            if bubble.follow_vehicle_id is not None:
                vehicle = self._last_vehicle_index.vehicle_by_id(
                    bubble.follow_vehicle_id, None
                )
                if vehicle is not None:
                    vehicles += [vehicle]
            assert (
                len(vehicles) <= 1
            ), "Travelling bubbles only support pinning to a single vehicle"

            if len(vehicles) == 1:
                bubble.move_to_follow_vehicle(vehicles[0])

    def _airlock_social_vehicle_with_social_agent(
        self, sim, vehicle_id: str, social_agent_actor: SocialAgentActor, bubble: Bubble
    ):
        """When airlocked. The social agent will receive observations and execute
        its policy, however it won't actually operate the vehicle's controller.
        """
        self._log.debug(
            f"Airlocked vehicle={vehicle_id} with actor={social_agent_actor}"
        )

        if bubble.is_boid:
            agent_id = BubbleManager._make_boid_social_agent_id(social_agent_actor)
        else:
            agent_id = BubbleManager._make_social_agent_id(vehicle_id)

        social_agent = None
        if (
            bubble.is_boid
            and bubble.keep_alive
            or agent_id in sim.agent_manager.social_agent_ids
        ):
            # E.g. if agent is a boid and was being re-used
            interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        else:
            social_agent = make_social_agent(
                locator=social_agent_actor.agent_locator,
                **social_agent_actor.policy_kwargs,
            )
            interface = social_agent.interface

        self._prepare_sensors_for_agent_control(
            sim, vehicle_id, agent_id, interface, bubble
        )

        if social_agent is None:
            return

        self._start_social_agent(
            sim, agent_id, social_agent, social_agent_actor, bubble
        )

    def _hijack_social_vehicle_with_social_agent(
        self, sim, vehicle_id: str, social_agent_actor: SocialAgentActor, bubble: Bubble
    ):
        """Upon hijacking the social agent is now in control of the vehicle. It will
        initialize the vehicle chassis (and by extension the controller) with a
        "greatest common denominator" state; that is: what's available via the vehicle
        front-end common to both source and destination policies during airlock.
        """
        self._log.debug(f"Hijack vehicle={vehicle_id} with actor={social_agent_actor}")
        if bubble.is_boid:
            agent_id = BubbleManager._make_boid_social_agent_id(social_agent_actor)
        else:
            agent_id = BubbleManager._make_social_agent_id(vehicle_id)

        try:
            agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        except KeyError:
            return
        vehicle = sim.vehicle_index.switch_control_to_agent(
            sim,
            vehicle_id,
            agent_id,
            boid=bubble.is_boid,
            hijacking=True,
            recreate=False,
            agent_interface=agent_interface,
        )
        sim.create_vehicle_in_providers(vehicle, agent_id)

    def _prepare_sensors_for_agent_control(
        self, sim, vehicle_id, agent_id, agent_interface, bubble
    ):
        plan = Plan(sim.road_map, None, find_route=False)
        vehicle = sim.vehicle_index.start_agent_observation(
            sim,
            vehicle_id,
            agent_id,
            agent_interface,
            plan,
            boid=bubble.is_boid,
        )

        # Setup mission (also used for observations)
        # XXX:  here we try to find where the vehicle was originally going, although
        # the agent may or may not want to go there too.  But we preserve it
        # in the plan so when the agent relinquishes control, the next Provider
        # can resume going there (potentially via a different route at that point).
        dest_road_id = None
        for traffic_sim in sim.traffic_sims:
            if traffic_sim.manages_actor(vehicle.id):
                dest_road_id = traffic_sim.vehicle_dest_road(vehicle.id)
                if dest_road_id is not None:
                    break
        if dest_road_id:
            goal = PositionalGoal.from_road(dest_road_id, sim.scenario.road_map)
        else:
            goal = EndlessGoal()
        mission = Mission(start=Start(vehicle.position[:2], vehicle.heading), goal=goal)
        try:
            plan.create_route(mission)
        except PlanningError:
            plan.route = sim.road_map.empty_route()

    def _start_social_agent(
        self, sim, agent_id, social_agent, social_agent_actor, bubble
    ):
        social_agent_data_model = SocialAgent(
            id=SocialAgentId.new(social_agent_actor.name),
            name=social_agent_actor.name,
            is_boid=bubble.is_boid,
            is_boid_keep_alive=bubble.keep_alive,
            agent_locator=social_agent_actor.agent_locator,
            policy_kwargs=social_agent_actor.policy_kwargs,
            initial_speed=social_agent_actor.initial_speed,
        )
        sim.agent_manager.start_social_agent(
            agent_id, social_agent, social_agent_data_model
        )

    @staticmethod
    def _make_social_agent_id(vehicle_id):
        return f"BUBBLE-AGENT-{truncate(vehicle_id, 48)}"

    @staticmethod
    def _make_boid_social_agent_id(social_agent_actor):
        return f"BUBBLE-AGENT-{truncate(social_agent_actor.name, 48)}"

    def teardown(self):
        """Clean up internal state."""
        self._cursors = set()
        self._bubbles = []
