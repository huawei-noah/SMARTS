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
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from sys import maxsize
from typing import Dict, FrozenSet, Sequence, Set, Tuple

from shapely.affinity import rotate, translate
from shapely.geometry import CAP_STYLE, JOIN_STYLE, Point, Polygon

from smarts.core.data_model import SocialAgent
from smarts.core.mission_planner import Mission, MissionPlanner, Start
from smarts.core.scenario import PositionalGoal
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.core.utils.id import SocialAgentId
from smarts.core.utils.string import truncate
from smarts.core.vehicle import Vehicle, VehicleState
from smarts.core.vehicle_index import VehicleIndex
from smarts.sstudio.types import BoidAgentActor
from smarts.sstudio.types import Bubble as SSBubble
from smarts.sstudio.types import BubbleLimits, SocialAgentActor
from smarts.zoo.registry import make as make_social_agent


class BubbleTransition(Enum):
    # --> [  AirlockEntered  [  Entered  ]  Exited  ]  AirlockExited -->
    AirlockEntered = 0  # --> (Airlock)
    Entered = 1  # Airlock --> (Bubble)
    Exited = 2  # Bubble --> (Airlock)
    AirlockExited = 3  # Airlock -->


class BubbleState(Enum):
    InBubble = 0
    InAirlock = 1


class Bubble:
    """Wrapper around `sstudio.types.Bubble` to cache bubble geometry (and does some
    internal Bubble-related business logic).
    """

    def __init__(self, bubble: SSBubble, sumo_road_network: SumoRoadNetwork):
        geometry = bubble.zone.to_geometry(sumo_road_network)

        bubble_limit = (
            bubble.limit or BubbleLimits()
            if bubble.limit is None or isinstance(bubble.limit, BubbleLimits)
            else BubbleLimits(bubble.limit, bubble.limit + 1)
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

        self._bubble_heading = 0
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
        return self._exclusion_prefixes

    @property
    def id(self):
        return self._bubble.id

    @property
    def actor(self) -> SocialAgentActor:
        return self._bubble.actor

    @property
    def follow_actor_id(self) -> str:
        return self._bubble.follow_actor_id

    @property
    def limit(self):
        return self._limit

    @property
    def is_boid(self):
        return self._bubble.is_boid

    @property
    def keep_alive(self):
        return self._bubble.keep_alive

    # XXX: In the case of travelling bubbles, the geometry and zone are moving
    #      according to the follow vehicle.
    @property
    def geometry(self) -> Polygon:
        return self._cached_inner_geometry

    @property
    def airlock_geometry(self) -> Polygon:
        return self._cached_airlock_geometry

    def admissibility(
        self,
        vehicle_id: str,
        index: VehicleIndex,
        vehicle_ids_in_bubbles: Dict["Bubble", Set[str]],
        running_cursors: Set["Cursor"],
    ):
        """The vehicle_id we are querying for and the `other_vehicle_ids` _presently in
        this `sstudio.types.Bubble`_.
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

            all_hijacked_vehicle_ids = (
                current_hijacked_vehicle_ids
                | vehicle_ids_by_bubble_state[BubbleState.InAirlock][self]
            ) - {vehicle_id}

            all_shadowed_vehicle_ids = (
                current_shadowed_vehicle_ids
                | vehicle_ids_by_bubble_state[BubbleState.InBubble][self]
            ) - {vehicle_id}

            hijackable = len(all_hijacked_vehicle_ids) < (
                self._limit.hijack_limit or maxsize
            )
            shadowable = len(all_shadowed_vehicle_ids) + len(
                all_hijacked_vehicle_ids
            ) < (self._limit.shadow_limit or maxsize)

        return hijackable, shadowable

    def in_bubble_or_airlock(self, position):
        if not isinstance(position, Point):
            position = Point(position)

        in_airlock = position.within(self._cached_airlock_geometry)
        if not in_airlock:
            return False, False

        in_bubble = position.within(self._cached_inner_geometry)
        return in_bubble, in_airlock and not in_bubble

    @property
    def is_travelling(self):
        return self._bubble.follow_actor_id is not None

    def move_to_follow_vehicle(self, vehicle: Vehicle):
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
)"""


@dataclass(frozen=True, eq=True)
class Cursor:
    """Tracks a vehicle through an airlock or a bubble."""

    # We would always want to have the vehicle go through the airlock zone. This may
    # not be the case if we spawn a vehicle in a bubble, but that wouldn't be ideal.
    vehicle_id: str
    state: BubbleState = None
    transition: BubbleTransition = None
    bubble: Bubble = None

    @classmethod
    def from_pos(
        cls,
        pos,
        vehicle,
        bubble,
        index,
        vehicle_ids_per_bubble: Dict[Bubble, Set[str]],
        running_cursors: Set["Cursor"],
    ):
        in_bubble_zone, in_airlock_zone = bubble.in_bubble_or_airlock(pos)
        is_social = vehicle.id in index.social_vehicle_ids()
        is_hijacked, is_shadowed = index.vehicle_is_hijacked_or_shadowed(vehicle.id)
        is_hijack_admissible, is_airlock_admissible = bubble.admissibility(
            vehicle.id, index, vehicle_ids_per_bubble, running_cursors
        )
        was_in_this_bubble = vehicle.id in vehicle_ids_per_bubble[bubble]

        # XXX: When a travelling bubble disappears and an agent is airlocked or
        #      hijacked. It remains in that state.
        # TODO: Depending on step size, we could potentially skip transitions (e.g.
        #       go straight to relinquish w/o hijacking first). This may be solved by
        #       time-based airlocking. For robust code we'll want to handle these
        #       scenarios (e.g. hijacking if didn't airlock first)
        transition = None
        if is_social and not is_shadowed and is_airlock_admissible and in_airlock_zone:
            transition = BubbleTransition.AirlockEntered
        elif is_shadowed and is_hijack_admissible and in_bubble_zone:
            transition = BubbleTransition.Entered
        elif is_hijacked and in_airlock_zone:
            # XXX: This may get called repeatedly because we don't actually change
            #      any state when this happens.
            transition = BubbleTransition.Exited
        elif (
            was_in_this_bubble
            and (is_shadowed or is_hijacked)
            and not (in_airlock_zone or in_bubble_zone)
        ):
            transition = BubbleTransition.AirlockExited

        state = None
        if is_hijack_admissible and in_bubble_zone:
            state = BubbleState.InBubble
        elif is_airlock_admissible and in_airlock_zone:
            state = BubbleState.InAirlock

        return cls(
            vehicle_id=vehicle.id, transition=transition, state=state, bubble=bubble
        )

    def __repr__(self):
        return f"Cursor(state={self.state}, transition={self.transition})"


class BubbleManager:
    def __init__(self, bubbles: Sequence[SSBubble], road_network: SumoRoadNetwork):
        self._log = logging.getLogger(self.__class__.__name__)
        self._cursors = set()
        self._last_vehicle_index = VehicleIndex.identity()
        self._bubbles = [Bubble(b, road_network) for b in bubbles]

    @property
    def bubbles(self) -> Sequence[Bubble]:
        return self._active_bubbles()

    def _active_bubbles(self) -> Sequence[Bubble]:
        # Filter out travelling bubbles that are missing their follow vehicle
        def is_active(bubble):
            if not bubble.is_travelling:
                return True

            vehicles = self._last_vehicle_index.vehicles_by_actor_id(
                bubble.follow_actor_id
            )
            return len(vehicles) == 1

        return [bubble for bubble in self._bubbles if is_active(bubble)]

    @staticmethod
    def _vehicle_ids_divided_by_bubble_state(
        cursors: FrozenSet[Cursor],
    ) -> Dict[Bubble, Set[str]]:
        vehicle_ids_grouped_by_cursor = defaultdict(lambda: defaultdict(set))
        for cursor in cursors:
            vehicle_ids_grouped_by_cursor[cursor.state][cursor.bubble].add(
                cursor.vehicle_id
            )
        return vehicle_ids_grouped_by_cursor

    @classmethod
    @lru_cache(maxsize=2)
    def vehicle_ids_per_bubble(
        cls,
        cursors: FrozenSet[Cursor],
    ) -> Dict[Bubble, Set[str]]:
        vid = cls._vehicle_ids_divided_by_bubble_state(cursors)
        return defaultdict(
            set, {**vid[BubbleState.InBubble], **vid[BubbleState.InAirlock]}
        )

    def step(self, sim):
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
        index_new = vehicle_index & last_vehicle_index

        # Calculate latest cursors
        vehicle_ids_per_bubble = BubbleManager.vehicle_ids_per_bubble(
            frozenset(self._cursors)
        )
        cursors = set()
        for _, vehicle in index_new.vehicleitems():
            # XXX: Turns out Point(...) creation is very expensive (~0.02ms) which
            #      when inside of a loop x large number of vehicles makes a big
            #      performance hit.
            point = Point(vehicle.position)
            for bubble in self._active_bubbles():
                cursors.add(
                    Cursor.from_pos(
                        pos=point,
                        vehicle=vehicle,
                        bubble=bubble,
                        index=vehicle_index,
                        vehicle_ids_per_bubble=vehicle_ids_per_bubble,
                        running_cursors=cursors,
                    )
                )

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
                if sim.vehicle_index.vehicle_is_hijacked(cursor.vehicle_id):
                    self._relinquish_vehicle_to_traffic_sim(
                        sim, cursor.vehicle_id, cursor.bubble
                    )
                else:
                    self._stop_shadowing_vehicle(sim, cursor.vehicle_id, cursor.bubble)

    def _move_travelling_bubbles(self, sim):
        for bubble in self._active_bubbles():
            if not bubble.is_travelling:
                continue

            vehicles = sim.vehicle_index.vehicles_by_actor_id(bubble.follow_actor_id)
            assert (
                len(vehicles) <= 1
            ), "Travelling bubbles only support pinning to a single vehicle"

            if len(vehicles) == 1:
                bubble.move_to_follow_vehicle(vehicles[0])

    def _airlock_social_vehicle_with_social_agent(
        self, sim, vehicle_id: str, social_agent_actor: SocialAgentActor, bubble: Bubble
    ) -> str:
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
    ) -> str:
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

        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        vehicle = sim.vehicle_index.switch_control_to_agent(
            sim,
            vehicle_id,
            agent_id,
            boid=bubble.is_boid,
            hijacking=True,
            recreate=False,
            agent_interface=agent_interface,
        )

        for provider in sim.providers:
            interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
            if interface.action_space in provider.action_spaces:
                provider.create_vehicle(
                    VehicleState(
                        vehicle_id=vehicle_id,
                        vehicle_type="passenger",
                        pose=vehicle.pose,
                        dimensions=vehicle.chassis.dimensions,
                        speed=vehicle.speed,
                        source="HIJACK",
                    )
                )

    def _relinquish_vehicle_to_traffic_sim(self, sim, vehicle_id: str, bubble: Bubble):
        agent_id = sim.vehicle_index.actor_id_from_vehicle_id(vehicle_id)
        shadow_agent_id = sim.vehicle_index.shadow_actor_id_from_vehicle_id(vehicle_id)

        # TODO: Remove this assumption. Elsewhere in the code, social_vehicle_id does
        #       not have to be equal to vehicle_id.
        social_vehicle_id = vehicle_id
        self._log.debug(
            f"Relinquish vehicle={vehicle_id} to traffic simulation (agent={agent_id}) "
            f"shadow_agent={shadow_agent_id} sv_id={social_vehicle_id})"
        )

        sim.vehicle_index.stop_agent_observation(vehicle_id)
        sim.vehicle_index.relinquish_agent_control(sim, vehicle_id, social_vehicle_id)
        if bubble.is_boid and bubble.keep_alive:
            return

        teardown_agent_ids = [agent_id] + ([shadow_agent_id] if shadow_agent_id else [])
        sim.teardown_agents_without_vehicles(teardown_agent_ids)

    def _stop_shadowing_vehicle(self, sim, vehicle_id: str, bubble: Bubble):
        shadow_agent_id = sim.vehicle_index.shadow_actor_id_from_vehicle_id(vehicle_id)
        self._log.debug(
            f"Stop shadowing vehicle={vehicle_id} (shadow_agent={shadow_agent_id})"
        )

        sim.vehicle_index.stop_agent_observation(vehicle_id)

        if bubble.is_boid and bubble.keep_alive:
            return

        sim.teardown_agents_without_vehicles([shadow_agent_id])

    def _prepare_sensors_for_agent_control(
        self, sim, vehicle_id, agent_id, agent_interface, bubble
    ):
        mission_planner = MissionPlanner(
            sim.scenario.waypoints, sim.scenario.road_network
        )
        vehicle = sim.vehicle_index.start_agent_observation(
            sim,
            vehicle_id,
            agent_id,
            agent_interface,
            mission_planner,
            boid=bubble.is_boid,
        )

        # Setup mission (also used for observations)
        route = sim.traffic_sim.vehicle_route(vehicle_id=vehicle.id)
        mission = Mission(
            start=Start(vehicle.position[:2], vehicle.heading),
            goal=PositionalGoal.fromedge(route[-1], sim.scenario.road_network),
        )
        mission_planner.plan(mission=mission)

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
        return f"BUBBLE-AGENT-{truncate(social_agent_actor.id, 48)}"

    def teardown(self):
        self._cursors = []
        self._bubbles = []
