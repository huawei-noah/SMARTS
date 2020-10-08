import logging
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Sequence, Tuple, Dict

from shapely.geometry import Polygon, Point

from smarts.core.data_model import SocialAgent
from smarts.core.mission_planner import Mission, MissionPlanner, Start
from smarts.core.scenario import PositionalGoal
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.core.vehicle import Vehicle, VehicleState
from smarts.sstudio.types import Bubble as SSBubble, BoidAgentActor
from smarts.sstudio.types import SocialAgentActor
from smarts.zoo.registry import make as make_social_agent


class BubbleState(Enum):
    # --> [  AirlockEntered  [  Entered  ]  Exited  ]  AirlockExited -->
    AirlockEntered = 0  # --> (Airlock)
    Entered = 1  # Airlock --> (Bubble)
    Exited = 2  # Bubble --> (Airlock)
    AirlockExited = 3  # Airlock -->


class Bubble:
    """Adhears to the same interface as `sstudio.types.Bubble` but caches the bubble
    geometry (and does some internal Bubble-related business logic).
    """

    def __init__(self, bubble: SSBubble, sumo_road_network: SumoRoadNetwork):
        geometry = bubble.zone.to_geometry(sumo_road_network)

        self._bubble = bubble
        self._limit = bubble.limit
        self._cached_inner_geometry = geometry
        self._exclusion_prefixes = bubble.exclusion_prefixes

        if isinstance(bubble.actor, BoidAgentActor):
            if bubble.limit is None:
                self._limit = bubble.actor.capacity
            elif bubble.actor.capacity is not None:
                self._limit = min(bubble.limit, bubble.actor.capacity)

        self._cached_airlock_geometry = self._cached_inner_geometry.buffer(
            bubble.margin
        )

    @property
    def geometry(self) -> Polygon:
        return self._cached_inner_geometry

    @property
    def exclusion_prefixes(self):
        return self._exclusion_prefixes

    @property
    def id(self):
        return self._bubble.id

    @property
    def zone(self):
        return self._bubble.zone

    @property
    def actor(self) -> SocialAgentActor:
        return self._bubble.actor

    @property
    def limit(self):
        self._limit

    @property
    def airlock_geometry(self) -> Polygon:
        return self._cached_airlock_geometry

    def is_admissible(self, vehicle_id: str, other_vehicle_ids: Sequence[int]):
        """The vehicle_id we are querying for and the other_vehicle_ids _presently in
        this `sstudio.types.Bubble_`.
        """

        for prefix in self.exclusion_prefixes:
            if vehicle_id.startswith(prefix):
                return False

        if self._limit is not None and len(other_vehicle_ids) >= self._limit:
            return False

        return True


@dataclass
class Cursor:
    """Tracks a vehicle through an airlock or a bubble."""

    # We would always want to have the vehicle go through the airlock zone. This may
    # not be the case if we spawn a vehicle in a bubble, but that wouldn't be ideal.
    vehicle: Vehicle
    state: BubbleState = None
    bubble: Bubble = None
    tracking_id: str = field(init=False)

    def __post_init__(self):
        self.tracking_id = self.vehicle.id

    def update(self) -> bool:
        """Going through an airlock transitions a vehicle between traffic control and
        agent control (into bubble; vice-versa out of bubble).

        Returns `True` on transition (entered or exited zone)
        """
        pos = Point(self.vehicle.position)

        state = self.state
        if state == BubbleState.AirlockEntered and pos.within(self.bubble.geometry):
            state = BubbleState.Entered
        elif state is None and pos.within(self.bubble.airlock_geometry):
            state = BubbleState.AirlockEntered
        elif state == BubbleState.Entered and not pos.within(self.bubble.geometry):
            state = BubbleState.Exited
        elif state == BubbleState.Exited and not pos.within(
            self.bubble.airlock_geometry
        ):
            state = BubbleState.AirlockExited
        elif state == BubbleState.AirlockEntered and not pos.within(
            self.bubble.airlock_geometry
        ):
            # Vehicle entered and exited the airlock region without going through the
            # inner bubble
            state = BubbleState.AirlockExited

        state_changed = state != self.state
        self.state = state
        return state_changed

    def __repr__(self):
        return f"""Cursor(
  tracking_id={self.tracking_id},
  vehicle={self.vehicle.id},
  state={self.state},
)"""


@dataclass(frozen=True)
class BubbleStateChange:
    # [(<vehicle_id>, <social_agent_actor>), ...]
    entered_airlock_1: Sequence[Tuple[str, SocialAgentActor]]

    # [(<vehicle_id>, <social_agent_actor>), ...]
    entered_bubble: Sequence[Tuple[str, SocialAgentActor]]

    # [<agent_id>, ...]
    exited_bubble: Sequence[str]

    # [<agent_id>, ...]
    exited_airlock_2: Sequence[str]


class BubbleManager:
    def __init__(self, bubbles: Sequence[SSBubble], road_network: SumoRoadNetwork):
        self._log = logging.getLogger(self.__class__.__name__)
        self._cursors = []
        self._bubbles = [Bubble(b, road_network) for b in bubbles]

    @property
    def bubbles(self) -> Sequence[Bubble]:
        return self._bubbles

    @property
    def tracked_vehicle_ids(self):
        return {c.vehicle.id for c in self._cursors}

    def vehicle_ids_in_bubble(self, bubble: SSBubble):
        bubble = self._internal_bubble_for_ss_bubble(bubble)
        cursors = self._group_cursors_by_bubble()[bubble]
        return [c.vehicle.id for c in cursors]

    def _internal_bubble_for_ss_bubble(self, bubble: SSBubble):
        for bubble_ in self._bubbles:
            if bubble_.id == bubble.id:
                return bubble_

        raise ValueError("Bubble does not exist")

    def forget_vehicles(self, vehicle_ids):
        """Removes any tracking information for the given vehicle ids"""
        self._cursors = [c for c in self._cursors if c.vehicle.id not in vehicle_ids]

    def step(self, sim):
        social_agent_vehicles = []
        for agent_id in sim.agent_manager.social_agent_ids:
            social_agent_vehicles += sim.vehicle_index.vehicles_by_actor_id(agent_id)

        social_vehicles = [
            sim.vehicle_index.vehicle_by_id(id_)
            for id_ in sim.vehicle_index.social_vehicle_ids
        ]
        state_change = self.step_bubble_state(social_vehicles, social_agent_vehicles)

        for vehicle_id, actor in state_change.entered_airlock_1:
            self._airlock_social_vehicle_with_social_agent(sim, vehicle_id, actor)

        for vehicle_id, actor in state_change.entered_bubble:
            self._hijack_social_vehicle_with_social_agent(sim, vehicle_id, actor)

        # XXX: Some vehicles only go through the airlocks and never make it through
        #      the bubble; that's why we relinquish on airlock exit. This is something
        #      we'll likely want to revisit in the future.
        for vehicle_id in state_change.exited_airlock_2:
            self._relinquish_vehicle_to_traffic_sim(sim, vehicle_id)

    def step_bubble_state(
        self,
        social_vehicles: Sequence[Vehicle],
        social_agent_vehicles: Sequence[Vehicle],
    ) -> BubbleStateChange:
        # Detect social vehicles entering bubbles (and airlocks)
        for sv in social_vehicles:
            cursor = self._find_cursor(sv.id)
            if cursor is not None:
                # Already tracking this vehicle (in airlock)
                continue

            # We didn't have a cursor for this vehicle yet
            cursor = self._obtain_cursor_if_entered_airlock(sv)
            if cursor is not None:
                self._cursors.append(cursor)

        kept_cursors = []
        entered_airlock_1, entered_bubble = [], []
        exited_bubble, exited_airlock_2 = [], []

        for cursor in self._cursors:
            if self._cursor_points_to_destroyed_vehicle(
                cursor, social_vehicles, social_agent_vehicles
            ):
                self._log.debug(
                    f"cursor={cursor.tracking_id} is pointing to a destroyed vehicle; "
                    "dropping it"
                )
                continue

            if not cursor.update():
                kept_cursors.append(cursor)
                continue

            if cursor.state == BubbleState.AirlockExited:
                self._log.debug(
                    f"vehicle={cursor.tracking_id} exited bubble and airlock"
                )
                exited_airlock_2.append(cursor.tracking_id)
                continue
            elif cursor.state == BubbleState.AirlockEntered:
                self._log.debug(f"vehicle={cursor.tracking_id} entered airlock")
                entered_airlock_1.append((cursor.tracking_id, cursor.bubble.actor))
            elif cursor.state == BubbleState.Entered:
                self._log.debug(f"vehicle={cursor.tracking_id} enter bubble")
                entered_bubble.append((cursor.tracking_id, cursor.bubble.actor))
            elif cursor.state == BubbleState.Exited:
                exited_bubble.append(cursor.tracking_id)
                self._log.debug(f"vehicle={cursor.tracking_id} exit bubble")

            kept_cursors.append(cursor)

        self._cursors = kept_cursors
        return BubbleStateChange(
            entered_airlock_1=entered_airlock_1,
            entered_bubble=entered_bubble,
            exited_bubble=exited_bubble,
            exited_airlock_2=exited_airlock_2,
        )

    def _airlock_social_vehicle_with_social_agent(
        self, sim, vehicle_id: str, social_agent_actor: SocialAgentActor,
    ) -> str:
        """When airlocked. The social agent will receive observations and execute
        its policy, however it won't actually operate the vehicle's controller.
        """
        self._log.debug(
            f"Airlocked vehicle={vehicle_id} with actor={social_agent_actor}"
        )

        agent_id = BubbleManager._make_social_agent_id(vehicle_id, social_agent_actor)

        if agent_id in sim.agent_manager.social_agent_ids:
            # E.g. if agent is a boid and was being re-used
            interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        else:
            social_agent = make_social_agent(
                locator=social_agent_actor.agent_locator,
                **social_agent_actor.policy_kwargs,
            )
            interface = social_agent.interface

        mission_planner = MissionPlanner(
            sim.scenario.waypoints, sim.scenario.road_network
        )
        vehicle = sim.vehicle_index.prepare_for_agent_control(
            sim, vehicle_id, agent_id, interface, mission_planner
        )

        # Setup mission (also used for observations)
        route = sim.traffic_sim.vehicle_route(vehicle_id=vehicle.id)
        mission = Mission(
            start=Start(vehicle.position[:2], vehicle.heading),
            goal=PositionalGoal.fromedge(route[-1], sim.scenario.road_network),
        )
        mission_planner.plan(mission=mission)

        if agent_id not in sim.agent_manager.social_agent_ids:
            social_agent_data_model = SocialAgent(
                id=f"social-agent-{social_agent_actor.name}",
                name=social_agent_actor.name,
                mission=mission,
                agent_locator=social_agent_actor.agent_locator,
                policy_kwargs=social_agent_actor.policy_kwargs,
                initial_speed=social_agent_actor.initial_speed,
            )
            sim.agent_manager.start_social_agent(
                agent_id, social_agent, social_agent_data_model
            )

        return agent_id

    def _hijack_social_vehicle_with_social_agent(
        self, sim, vehicle_id: str, social_agent_actor: SocialAgentActor,
    ) -> str:
        """Upon hijacking the social agent is now in control of the vehicle. It will
        initialize the vehicle chassis (and by extension the controller) with a
        "greatest common denominator" state; that is: what's available via the vehicle
        front-end common to both source and destination policies during airlock.
        """
        self._log.debug(f"Hijack vehicle={vehicle_id} with actor={social_agent_actor}")
        agent_id = BubbleManager._make_social_agent_id(vehicle_id, social_agent_actor)

        vehicle = sim.vehicle_index.switch_control_to_agent(
            sim, vehicle_id, agent_id, recreate=False
        )
        self._update_cursor(vehicle_id, vehicle=vehicle)

        for provider in sim.providers:
            if (
                sim.agent_manager.agent_interface_for_agent_id(agent_id).action_space
                in provider.action_spaces
            ):
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

    def _relinquish_vehicle_to_traffic_sim(self, sim, vehicle_id: str) -> str:
        agent_id = sim.vehicle_index.actor_id_from_vehicle_id(vehicle_id)
        shadow_agent_id = sim.vehicle_index.shadow_actor_id_from_vehicle_id(vehicle_id)

        # TODO: Remove this assumption. Elsewhere in the code, social_vehicle_id does
        #       not have to be equal to vehicle_id.
        social_vehicle_id = vehicle_id
        self._log.debug(
            f"Relinquish vehicle={vehicle_id} to traffic simulation (agent={agent_id} "
            f"shadow_agent={shadow_agent_id} sv_id={social_vehicle_id})"
        )
        sim.vehicle_index.relinquish_agent_control(sim, vehicle_id, social_vehicle_id)
        sim.teardown_agents_without_vehicles([agent_id, shadow_agent_id])
        return social_vehicle_id

    @staticmethod
    def _make_social_agent_id(vehicle_id, social_agent_actor):
        if isinstance(social_agent_actor, BoidAgentActor):
            return social_agent_actor.id

        return f"BUBBLE-AGENT-{vehicle_id}"

    def _obtain_cursor_if_entered_airlock(self, vehicle: Vehicle):
        """If a vehicle entered an airlock we return a corresponding cursor. We do not
        handle overlapping airlocks/bubbles, but only return a cursor for the first
        according to the provided order.
        """
        pos = Point(vehicle.position)
        for bubble in self._bubbles:
            all_bubble_vehicle_ids = self._group_cursors_by_bubble()[bubble]
            # Admissibility needs to be considered upon Bubble entry since some
            # admission reasons (like bubble capacity/limit) could change at any time,
            # but we would not want this vehicle in question to "pop in" to the bubble.
            # If the count is not below the limit before entry the entering vehicle
            # lost its chance to be hijacked.
            within_airlock_geometry = pos.within(
                bubble.airlock_geometry
            ) and not pos.within(bubble.geometry)
            if (
                bubble.is_admissible(vehicle.id, all_bubble_vehicle_ids)
                and within_airlock_geometry
            ):
                return Cursor(vehicle=vehicle, bubble=bubble)

        return None  # not in bubble

    def _update_cursor(self, tracking_id: str, **kwargs):
        for i in range(len(self._cursors)):
            if self._cursors[i].tracking_id == tracking_id:
                self._cursors[i] = replace(self._cursors[i], **kwargs)

    def _find_cursor(self, tracking_id: str):
        for cursor in self._cursors:
            if cursor.tracking_id == tracking_id:
                return cursor

        return None  # not found

    def _group_cursors_by_bubble(self) -> Dict[Bubble, Cursor]:
        grouped = defaultdict(list)
        for cursor in self._cursors:
            grouped[cursor.bubble].append(cursor)

        return grouped

    def _cursor_points_to_destroyed_vehicle(
        self, cursor, social_vehicles, social_agent_vehicles
    ):
        social_agent_vehicle_ids = [v.id for v in social_agent_vehicles]
        destroyed_agent_vehicle = (
            cursor.state == BubbleState.Entered
            and cursor.tracking_id not in social_agent_vehicle_ids
        )
        if destroyed_agent_vehicle:
            self._log.debug(
                f"cursor={cursor.tracking_id} pointing to destroyed agent vehicle"
            )
            return True

        social_vehicle_ids = [v.id for v in social_vehicles]
        destroyed_social_vehicle = (
            cursor.state in (BubbleState.AirlockEntered, BubbleState.AirlockExited)
            and cursor.tracking_id not in social_vehicle_ids
        )
        if destroyed_social_vehicle:
            self._log.debug(
                f"cursor={cursor.tracking_id} pointing to destroyed social vehicle"
            )
            return True

        return False

    def teardown(self):
        self._cursors = []
        self._bubbles = []
