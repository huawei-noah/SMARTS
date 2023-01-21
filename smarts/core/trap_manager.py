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
import random as rand
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from shapely.geometry import Polygon

from smarts.core.coordinates import Point as MapPoint
from smarts.core.plan import Mission, Plan, Start, default_entry_tactic
from smarts.core.utils.file import replace
from smarts.core.utils.math import clip, squared_dist
from smarts.core.vehicle import Vehicle
from smarts.sstudio.types import MapZone, PositionalZone, TrapEntryTactic


@dataclass
class Trap:
    """Caches geometry and temporal information to use to capture actors for social agents"""

    geometry: Polygon
    """The trap area within which actors are considered for capture."""
    mission: Mission
    """The mission that this trap should assign the captured actor."""
    exclusion_prefixes: Sequence[str]
    """Prefixes of actors that should be ignored by this trap."""
    activation_time: float
    """The amount of time left until this trap activates."""
    patience: float
    """Patience to wait for better capture circumstances after which the trap expires."""
    default_entry_speed: float
    """The default entry speed of a new vehicle should this trap expire."""

    def ready(self, sim_time: float):
        """If the trap is ready to capture a vehicle."""
        return self.activation_time < sim_time or math.isclose(
            self.activation_time, sim_time
        )

    def patience_expired(self, sim_time: float):
        """If the trap has expired and should no longer capture a vehicle."""
        expiry_time = self.activation_time + self.patience
        return expiry_time < sim_time or math.isclose(expiry_time, sim_time)

    def includes(self, vehicle_id: str):
        """Returns if the given actor should be considered for capture."""
        for prefix in self.exclusion_prefixes:
            if vehicle_id.startswith(prefix):
                return False
        return True


class TrapManager:
    """Facilitates agent hijacking of actors"""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._traps: Dict[str, Trap] = {}
        self._cancelled_agents: Set[str] = set()

    def init_traps(self, road_map, missions, sim):
        """Set up the traps used to capture actors."""
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)
        self._traps.clear()
        for agent_id, mission in missions.items():
            self.add_trap_for_agent(
                agent_id, mission, road_map, sim.elapsed_sim_time, reject_expired=True
            )
        if len(self._cancelled_agents) > 0:
            sim.agent_manager.teardown_ego_agents(self._cancelled_agents)
            self._cancelled_agents.clear()

    def add_trap_for_agent(
        self,
        agent_id: str,
        mission: Mission,
        road_map,
        sim_time: float,
        reject_expired: bool = False,
    ) -> bool:
        """Add a new trap to capture an actor for the given agent.

        :param agent_id: The agent to associate to this trap.
        :type agent_id: str
        :param mission: The mission to assign to the agent and vehicle.
        :type mission: class: Mission
        :param road_map: The road map to provide information to about the map.
        :type road_map: class: RoadMap
        :param sim_time: The current simulator time.
        :type sim_time: float
        :param reject_expired: If traps should be ignored if their patience would already be
            expired on creation
        :type reject_expired: bool
        """
        if mission is None:
            mission = Mission.random_endless_mission(road_map)

        if not mission.entry_tactic:
            mission = replace(mission, entry_tactic=default_entry_tactic())

        if not isinstance(mission.entry_tactic, TrapEntryTactic):
            return False

        entry_tactic = mission.entry_tactic
        assert isinstance(entry_tactic, TrapEntryTactic)
        # Do not add trap if simulation time is specified and patience already expired
        patience_expired = mission.start_time + entry_tactic.wait_to_hijack_limit_s
        if reject_expired and patience_expired < sim_time:
            self._log.warning(
                f"Trap skipped for `{agent_id}` scheduled to start between "
                + f"`{mission.start_time}` and `{patience_expired}` because simulation skipped to "
                f"simulation time `{sim_time}`"
            )
            self._cancelled_agents.add(agent_id)
            return False

        plan = Plan(road_map, mission)
        trap = self._mission2trap(road_map, plan.mission)
        self._traps[agent_id] = trap
        return True

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

    def step(self, sim):
        """Run hijacking and update agent and actor states."""
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)
        captures_by_agent_id = defaultdict(list)

        # Do an optimization to only check if there are pending agents.
        if not sim.agent_manager.pending_agent_ids:
            return

        social_vehicle_ids: List[str] = [
            v_id
            for v_id in sim.vehicle_index.social_vehicle_ids()
            if not sim.vehicle_index.vehicle_is_shadowed(v_id)
        ]
        vehicles: Dict[str, Vehicle] = {
            v_id: sim.vehicle_index.vehicle_by_id(v_id) for v_id in social_vehicle_ids
        }

        def largest_vehicle_plane_dimension(vehicle: Vehicle):
            return max(*vehicle.chassis.dimensions.as_lwh[:2])

        vehicle_comp = [
            (v.position[:2], largest_vehicle_plane_dimension(v), v)
            for v in vehicles.values()
        ]

        for agent_id in sim.agent_manager.pending_agent_ids:
            trap = self._traps.get(agent_id)

            if trap is None:
                continue

            if not trap.ready(sim.elapsed_sim_time):
                continue

            # Order vehicle ids by distance.
            sorted_vehicle_ids = sorted(
                list(social_vehicle_ids),
                key=lambda v: squared_dist(
                    vehicles[v].position[:2], trap.mission.start.position[:2]
                ),
            )
            for v_id in sorted_vehicle_ids:
                # Skip the capturing process if history traffic is used
                if trap.mission.vehicle_spec is not None:
                    break

                if not trap.includes(v_id):
                    continue

                vehicle: Vehicle = vehicles[v_id]
                point = vehicle.pose.point.as_shapely

                if not point.within(trap.geometry):
                    continue

                captures_by_agent_id[agent_id].append(
                    (
                        v_id,
                        trap,
                        replace(
                            trap.mission,
                            start=Start(vehicle.position[:2], vehicle.pose.heading),
                        ),
                    )
                )
                # TODO: Resolve overlap using a tree instead of just removing.
                social_vehicle_ids.remove(v_id)
                break

        # Use fed in trapped vehicles.
        agents_given_vehicle = set()
        used_traps = []
        for agent_id in sim._agent_manager.pending_agent_ids:
            if agent_id not in self._traps:
                continue

            trap = self._traps[agent_id]

            captures = captures_by_agent_id[agent_id]

            if not trap.ready(sim.elapsed_sim_time):
                continue

            vehicle = None
            if len(captures) > 0:
                vehicle_id, trap, mission = rand.choice(captures)
                vehicle = sim.switch_control_to_agent(
                    vehicle_id, agent_id, mission, recreate=True, is_hijacked=False
                )
            elif trap.patience_expired(sim.elapsed_sim_time):
                # Make sure there is not a vehicle in the same location
                mission = trap.mission
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

                vehicle = TrapManager._make_vehicle(
                    sim, agent_id, mission, trap.default_entry_speed
                )
            else:
                continue
            if vehicle == None:
                continue
            sim.create_vehicle_in_providers(vehicle, agent_id, True)
            agents_given_vehicle.add(agent_id)
            used_traps.append((agent_id, trap))

        if len(agents_given_vehicle) > 0:
            self.remove_traps(used_traps)
            sim.agent_manager.remove_pending_agent_ids(agents_given_vehicle)

    @property
    def traps(self) -> Dict[str, Trap]:
        """The traps in this manager."""
        return self._traps

    @staticmethod
    def _make_vehicle(sim, agent_id, mission, initial_speed):
        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        plan = Plan(sim.road_map, mission)
        # 3. Apply agent vehicle association.
        vehicle = sim.vehicle_index.build_agent_vehicle(
            sim,
            agent_id,
            agent_interface,
            plan,
            sim.scenario.vehicle_filepath,
            sim.scenario.tire_parameters_filepath,
            True,
            sim.scenario.surface_patches,
            initial_speed=initial_speed,
            boid=False,
        )
        return vehicle

    def reset(self):
        """Resets to a pre-initialized state."""
        self.captures_by_agent_id = defaultdict(list)

    def teardown(self):
        """Clear internal state"""
        self.reset()
        self._traps.clear()

    def _mission2trap(self, road_map, mission: Mission, default_zone_dist: float = 6.0):
        if not (hasattr(mission, "start") and hasattr(mission, "goal")):
            raise ValueError(f"Value {mission} is not a mission!")

        assert isinstance(mission.entry_tactic, TrapEntryTactic)

        patience = mission.entry_tactic.wait_to_hijack_limit_s
        zone = mission.entry_tactic.zone
        default_entry_speed = mission.entry_tactic.default_entry_speed
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
            exclusion_prefixes=mission.entry_tactic.exclusion_prefixes,
            default_entry_speed=default_entry_speed,
        )

        return trap
