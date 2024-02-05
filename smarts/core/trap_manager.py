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
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from shapely.geometry import Polygon

from smarts.core.actor_capture_manager import ActorCaptureManager
from smarts.core.condition_state import ConditionState
from smarts.core.coordinates import Point as MapPoint
from smarts.core.plan import NavigationMission, Plan, Start, default_entry_tactic
from smarts.core.utils.core_math import clip, squared_dist
from smarts.core.utils.file import replace
from smarts.core.vehicle import Vehicle
from smarts.sstudio.sstypes import MapZone, PositionalZone, TrapEntryTactic

if TYPE_CHECKING:
    import smarts.core.scenario
    from smarts.core.road_map import RoadMap
    from smarts.core.smarts import SMARTS


@dataclass
class Trap:
    """Caches geometry and temporal information to use to capture actors for social agents"""

    geometry: Polygon
    """The trap area within which actors are considered for capture."""
    mission: NavigationMission
    """The mission that this trap should assign the captured actor."""
    activation_time: float
    """The amount of time left until this trap activates."""
    patience: float
    """Patience to wait for better capture circumstances after which the trap expires."""
    default_entry_speed: float
    """The default entry speed of a new vehicle should this trap expire."""
    entry_tactic: TrapEntryTactic
    """The entry tactic that this trap was generated with."""

    def ready(self, sim_time: float):
        """If the trap is ready to capture a vehicle."""
        return self.activation_time < sim_time or math.isclose(
            self.activation_time, sim_time
        )

    def patience_expired(self, sim_time: float):
        """If the trap has expired and should no longer capture a vehicle."""
        expiry_time = self.activation_time + self.patience
        return expiry_time < sim_time and not math.isclose(expiry_time, sim_time)

    def includes(self, vehicle_id: str):
        """Returns if the given actor should be considered for capture."""
        for prefix in self.entry_tactic.exclusion_prefixes:
            if vehicle_id.startswith(prefix):
                return False
        return True


@dataclass(frozen=True)
class _CaptureState:
    ready_state: ConditionState
    trap: Optional[Trap]
    vehicle_id: Optional[str] = None
    updated_mission: Optional[NavigationMission] = None
    default: bool = False


class TrapManager(ActorCaptureManager):
    """Facilitates agent hijacking of actors"""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._traps: Dict[str, Trap] = {}

    def init_traps(self, road_map, missions, sim: SMARTS):
        """Set up the traps used to capture actors."""
        self._traps.clear()
        cancelled_agents: Set[str] = set()
        for agent_id, mission in missions.items():
            added, expired = self.add_trap_for_agent(
                agent_id, mission, road_map, sim.elapsed_sim_time, reject_expired=True
            )
            if expired and not added:
                cancelled_agents.add(agent_id)
        if len(cancelled_agents) > 0:
            sim.agent_manager.teardown_ego_agents(cancelled_agents)

    def add_trap_for_agent(
        self,
        agent_id: str,
        mission: NavigationMission,
        road_map: RoadMap,
        sim_time: float,
        reject_expired: bool = False,
    ) -> Tuple[bool, bool]:
        """Add a new trap to capture an actor for the given agent.

        :param agent_id: The agent to associate to this trap.
        :type agent_id: str
        :param mission: The mission to assign to the agent and vehicle.
        :type mission: smarts.core.plan.NavigationMission
        :param road_map: The road map to provide information to about the map.
        :type road_map: RoadMap
        :param sim_time: The current simulator time.
        :type sim_time: float
        :param reject_expired: If traps should be ignored if their patience would already be
            expired on creation
        :type reject_expired: bool
        :return: If the trap was added and if the trap is already expired.
        :rtype: Tuple[bool, bool]
        """
        if mission is None:
            mission = NavigationMission.random_endless_mission(road_map)

        if not mission.entry_tactic:
            mission = replace(mission, entry_tactic=default_entry_tactic())

        if not isinstance(mission.entry_tactic, TrapEntryTactic):
            return False, False

        entry_tactic: TrapEntryTactic = (
            mission.entry_tactic
        )  # pytype: disable=annotation-type-mismatch
        # Do not add trap if simulation time is specified and patience already expired
        patience_expired = mission.start_time + entry_tactic.wait_to_hijack_limit_s
        if reject_expired and patience_expired < sim_time:
            self._log.warning(
                f"Trap skipped for `{agent_id}` scheduled to start between "
                + f"`{mission.start_time}` and `{patience_expired}` because simulation skipped to "
                f"simulation time `{sim_time}`"
            )
            return False, True

        plan = Plan(road_map, mission)
        trap = self._mission2trap(road_map, plan.mission)
        self._traps[agent_id] = trap
        return True, False

    def remove_traps(self, used_traps):
        """Remove the given used traps."""
        for agent_id, _ in used_traps:
            del self._traps[agent_id]

    def reset_traps(self, used_traps):
        """Reset all used traps."""
        self._log.warning(
            "Please update usage: ",
            exc_info=DeprecationWarning(
                "`TrapManager.reset_traps(..)` method has been deprecated in favor of `remove_traps(..)`."
            ),
        )
        self.remove_traps(used_traps)

    def step(self, sim: SMARTS):
        """Run vehicle hijacking and update agent and actor states."""
        capture_by_agent_id: Dict[str, _CaptureState] = defaultdict(
            lambda: _CaptureState(
                ready_state=ConditionState.FALSE,
                trap=None,
                default=True,
            )
        )

        # An optimization to short circuit if there are no pending agents.
        if not (
            sim.agent_manager.pending_agent_ids
            | sim.agent_manager.pending_social_agent_ids
        ):
            return

        social_vehicle_ids: List[str] = [
            v_id
            for v_id in sim.vehicle_index.social_vehicle_ids()
            if not sim.vehicle_index.vehicle_is_shadowed(v_id)
        ]
        vehicles: Dict[str, Vehicle] = {
            v_id: sim.vehicle_index.vehicle_by_id(v_id) for v_id in social_vehicle_ids
        }

        vehicle_comp = [
            (v.position[:2], max(v.chassis.dimensions.as_lwh[:2]), v)
            for v in vehicles.values()
        ]

        pending_agent_ids = (
            sim.agent_manager.pending_agent_ids
            | sim.agent_manager.pending_social_agent_ids
        )
        # Pending agents is currently used to avoid
        for agent_id in pending_agent_ids:
            trap = self._traps.get(agent_id)

            if trap is None:
                continue

            # Skip the capturing process if history traffic is used
            if trap.mission.vehicle_spec is not None:
                continue

            if not trap.ready(sim.elapsed_sim_time):
                capture_by_agent_id[agent_id] = _CaptureState(
                    ConditionState.BEFORE, trap
                )
                continue

            if (
                trap.patience_expired(sim.elapsed_sim_time)
                and sim.elapsed_sim_time > sim.fixed_timestep_sec
            ):
                capture_by_agent_id[agent_id] = _CaptureState(
                    ConditionState.EXPIRED, trap, updated_mission=trap.mission
                )
                continue

            sim_eval_kwargs = ActorCaptureManager._gen_simulation_condition_kwargs(
                sim, trap.entry_tactic.condition.requires
            )
            mission_eval_kwargs = ActorCaptureManager._gen_mission_condition_kwargs(
                agent_id, trap.mission, trap.entry_tactic.condition.requires
            )
            trap_condition = trap.entry_tactic.condition.evaluate(
                **sim_eval_kwargs, **mission_eval_kwargs
            )
            if not trap_condition:
                capture_by_agent_id[agent_id] = _CaptureState(trap_condition, trap)
                continue

            # Order vehicle ids by distance.
            sorted_vehicle_ids = sorted(
                list(social_vehicle_ids),
                key=lambda v: squared_dist(
                    vehicles[v].position[:2], trap.mission.start.position[:2]
                ),
            )
            for vehicle_id in sorted_vehicle_ids:
                if not trap.includes(vehicle_id):
                    continue

                vehicle: Vehicle = vehicles[vehicle_id]
                point = vehicle.pose.point.as_shapely

                if not point.within(trap.geometry):
                    continue

                capture_by_agent_id[agent_id] = _CaptureState(
                    ready_state=trap_condition,
                    trap=trap,
                    updated_mission=replace(
                        trap.mission,
                        start=Start(vehicle.position[:2], vehicle.pose.heading),
                    ),
                    vehicle_id=vehicle_id,
                )
                social_vehicle_ids.remove(vehicle_id)
                break
            else:
                capture_by_agent_id[agent_id] = _CaptureState(
                    ready_state=trap_condition,
                    trap=trap,
                )

        used_traps = []
        for agent_id in pending_agent_ids:
            capture = capture_by_agent_id[agent_id]

            if capture.default:
                continue

            if capture.trap is None:
                continue

            if not capture.trap.ready(sim.elapsed_sim_time):
                continue

            vehicle: Optional[Vehicle] = None
            if capture.ready_state and capture.vehicle_id is not None:
                vehicle = self._take_existing_vehicle(
                    sim,
                    capture.vehicle_id,
                    agent_id,
                    capture.updated_mission,
                    social=agent_id in sim.agent_manager.pending_social_agent_ids,
                )
            elif ConditionState.EXPIRED in capture.ready_state:
                # Make sure there is not a vehicle in the same location
                mission = capture.updated_mission or capture.trap.mission
                if mission.vehicle_spec is None:
                    nv_dims = Vehicle.agent_vehicle_dims(mission)
                    new_veh_maxd = max(nv_dims.as_lwh[:2])
                    overlapping = False
                    for pos, largest_dimension, _ in vehicle_comp:
                        if (
                            squared_dist(pos, mission.start.position[:2])
                            <= (0.5 * (largest_dimension + new_veh_maxd)) ** 2
                        ):
                            overlapping = True
                            break
                    if overlapping:
                        continue

                vehicle = TrapManager._make_new_vehicle(
                    sim,
                    agent_id,
                    mission,
                    capture.trap.default_entry_speed,
                    social=agent_id in sim.agent_manager.pending_social_agent_ids,
                )
            else:
                continue
            if vehicle is None:
                continue
            used_traps.append((agent_id, capture.trap))

        self.remove_traps(used_traps)

    @property
    def traps(self) -> Dict[str, Trap]:
        """The traps in this manager."""
        return self._traps

    def reset(self, scenario: smarts.core.scenario.Scenario, sim: SMARTS):
        """
        :param scenario: The scenario to initialize from.
        :type scenario: smarts.core.scenario.Scenario
        :param sim: The simulation this is associated to.
        :type scenario: smarts.core.smarts.SMARTS
        """
        self.init_traps(scenario.road_map, scenario.missions, sim)

    def teardown(self):
        self._traps.clear()

    def _mission2trap(
        self,
        road_map: RoadMap,
        mission: NavigationMission,
        default_zone_dist: float = 6.0,
    ):
        if not (hasattr(mission, "start") and hasattr(mission, "goal")):
            raise ValueError(f"Value {mission} is not a mission!")

        entry_tactic = mission.entry_tactic
        assert isinstance(entry_tactic, TrapEntryTactic)

        patience = entry_tactic.wait_to_hijack_limit_s
        zone = entry_tactic.zone
        default_entry_speed = entry_tactic.default_entry_speed
        n_lane = None

        if default_entry_speed is None:
            n_lane = road_map.nearest_lane(mission.start.point)
            default_entry_speed = n_lane.speed_limit if n_lane is not None else 0

        if zone is None:
            n_lane = n_lane or road_map.nearest_lane(mission.start.point)
            if n_lane is None:
                zone = PositionalZone(mission.start.position[:2], size=(3, 3))
            else:
                lane_speed = n_lane.speed_limit
                start_road_id = n_lane.road.road_id
                start_lane = n_lane.index
                lane_length = n_lane.length
                start_pos = mission.start.position
                vehicle_offset_into_lane = n_lane.offset_along_lane(
                    MapPoint(x=start_pos[0], y=start_pos[1])
                )
                vehicle_offset_into_lane = clip(
                    vehicle_offset_into_lane, 1e-6, lane_length - 1e-6
                )

                drive_distance = lane_speed * default_zone_dist

                start_offset_in_lane = vehicle_offset_into_lane - drive_distance
                start_offset_in_lane = clip(
                    start_offset_in_lane, 1e-6, lane_length - 1e-6
                )
                length = max(1e-6, vehicle_offset_into_lane - start_offset_in_lane)

                zone = MapZone(
                    start=(start_road_id, start_lane, start_offset_in_lane),
                    length=length,
                    n_lanes=1,
                )

        trap = Trap(
            geometry=zone.to_geometry(road_map),
            activation_time=mission.start_time,
            patience=patience,
            mission=mission,
            default_entry_speed=default_entry_speed,
            entry_tactic=mission.entry_tactic,
        )

        return trap
