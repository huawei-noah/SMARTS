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
import random as rand
import logging

from collections import defaultdict
from typing import Dict, Sequence
from dataclasses import dataclass, replace

import numpy as np
from shapely.geometry import Point, Polygon

from smarts.core.mission_planner import Mission, MissionPlanner
from smarts.core.scenario import Start, default_entry_tactic
from smarts.core.trap_config import TrapConfig
from smarts.core.vehicle import VehicleState
from smarts.sstudio.types import MapZone, TrapEntryTactic
from smarts.core.utils.math import clip, squared_dist


@dataclass
class Trap:
    """Caches geometry and temporal information to use to capture social agents"""

    geometry: Polygon
    mission: Mission
    exclusion_prefixes: Sequence[str]
    reactivation_time: float
    remaining_time_to_reactivation: float
    patience: float

    def step_trigger(self, dt: float):
        self.remaining_time_to_reactivation -= dt

    @property
    def ready(self):
        return self.remaining_time_to_reactivation < 0

    @property
    def patience_expired(self):
        """Patience recommendation to wait for better capture circumstances"""
        return self.remaining_time_to_reactivation < -self.patience

    def reset_trigger(self):
        self.remaining_time_to_reactivation = self.reactivation_time

    @property
    def reactivates(self):
        """If the trap is reusable"""
        return self.reactivation_time >= 0

    def includes(self, vehicle_id: str):
        for prefix in self.exclusion_prefixes:
            if vehicle_id.startswith(prefix):
                return False
        return True


class TrapManager:
    """Facilitates ego hijacking of social vehicles"""

    def __init__(self, scenario):
        self._log = logging.getLogger(self.__class__.__name__)
        self._traps: Dict[Trap] = defaultdict(None)
        self.init_traps(scenario)

    def init_traps(self, scenario):
        self._traps.clear()
        trap_configs: Sequence[TrapConfig] = []

        for agent_id in scenario.missions:
            mission = scenario.missions[agent_id]
            mission_planner = MissionPlanner(scenario.waypoints, scenario.road_network)
            if mission is None:
                mission = mission_planner.random_endless_mission()

            if not mission.entry_tactic:
                mission = replace(mission, entry_tactic=default_entry_tactic())

            if (
                not isinstance(mission.entry_tactic, TrapEntryTactic)
                and mission.entry_tactic
            ):
                continue

            mission_planner.plan(mission)

            trap_config = self._mission2trap(scenario.road_network, mission)
            trap_configs.append((agent_id, trap_config))

        for agent_id, tc in trap_configs:
            trap = Trap(
                geometry=tc.zone.to_geometry(scenario.road_network),
                mission=tc.mission,
                exclusion_prefixes=tc.exclusion_prefixes,
                reactivation_time=tc.reactivation_time,
                remaining_time_to_reactivation=tc.activation_delay,
                patience=tc.patience,
            )
            self.add_trap_for_agent_id(agent_id, trap)

    def add_trap_for_agent_id(self, agent_id, trap: Trap):
        self._traps[agent_id] = trap

    def reset_traps(self, used_traps):
        for agent_id, trap in used_traps:
            trap.reset_trigger()
            if not trap.reactivates:
                del self._traps[agent_id]

    def step(self, sim):
        captures_by_agent_id = defaultdict(list)

        # Do an optimization to only check if there are pending agents.
        if not sim.agent_manager.pending_agent_ids:
            return

        social_vehicle_ids = sim.vehicle_index.social_vehicle_ids
        vehicles = {
            v_id: sim.vehicle_index.vehicle_by_id(v_id) for v_id in social_vehicle_ids
        }

        existing_agent_vehicles = (
            sim.vehicle_index.vehicle_by_id(v_id)
            for v_id in sim.vehicle_index.agent_vehicle_ids
        )

        def largest_vehicle_plane_dimension(vehicle):
            return max(*vehicle.chassis.dimensions.as_lwh[:2])

        agent_vehicle_comp = [
            (v.position[:2], largest_vehicle_plane_dimension(v), v)
            for v in existing_agent_vehicles
        ]

        for agent_id in sim.agent_manager.pending_agent_ids:
            trap = self._traps[agent_id]

            if trap is None:
                continue

            trap.step_trigger(sim.timestep_sec)

            if not trap.ready:
                continue

            # Order vehicle ids by distance.
            sorted_vehicle_ids = sorted(
                list(social_vehicle_ids),
                key=lambda v: squared_dist(
                    vehicles[v].position[:2], trap.mission.start.position
                ),
            )
            for v_id in sorted_vehicle_ids:
                vehicle = vehicles[v_id]
                point = Point(vehicle.position)

                if any(v_id.startswith(prefix) for prefix in trap.exclusion_prefixes):
                    continue

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

            if not trap.ready:
                continue

            vehicle = None
            if len(captures) > 0:
                vehicle_id, trap, mission = rand.choice(captures)
                vehicle = TrapManager._hijack_vehicle(
                    sim, vehicle_id, agent_id, mission
                )
            elif trap.patience_expired:
                if len(agent_vehicle_comp) > 0:
                    agent_vehicle_comp.sort(
                        key=lambda v: squared_dist(v[0], trap.mission.start.position)
                    )

                    # Make sure there is not an agent vehicle in the same location
                    pos, largest_dimension, _ = agent_vehicle_comp[0]
                    if (
                        squared_dist(pos, trap.mission.start.position)
                        < largest_dimension
                    ):
                        continue

                vehicle = TrapManager._make_vehicle(sim, agent_id, trap.mission)
            else:
                continue

            if vehicle == None:
                continue

            agents_given_vehicle.add(agent_id)
            used_traps.append((agent_id, trap))

            for provider in sim.providers:
                if (
                    sim.agent_manager.agent_interface_for_agent_id(
                        agent_id
                    ).action_space
                    in provider.action_spaces
                ):
                    provider.create_vehicle(
                        VehicleState(
                            vehicle_id=vehicle.id,
                            vehicle_type="passenger",
                            pose=vehicle.pose,
                            dimensions=vehicle.chassis.dimensions,
                            speed=vehicle.speed,
                            source="EGO-HIJACK",
                        )
                    )
        if len(agents_given_vehicle) > 0:
            self.reset_traps(used_traps)
            sim.agent_manager.remove_pending_agent_ids(agents_given_vehicle)

    @property
    def traps(self):
        return self._traps

    @staticmethod
    def _hijack_vehicle(sim, vehicle_id, agent_id, mission):
        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        planner = MissionPlanner(sim.scenario.waypoints, sim.scenario.road_network)
        planner.plan(mission=mission)

        # Apply agent vehicle association.
        sim.vehicle_index.prepare_for_agent_control(
            sim, vehicle_id, agent_id, agent_interface, planner
        )
        vehicle = sim.vehicle_index.switch_control_to_agent(
            sim, vehicle_id, agent_id, recreate=True, hijacking=False
        )
        return vehicle

    @staticmethod
    def _make_vehicle(sim, agent_id, mission):
        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        planner = MissionPlanner(sim.scenario.waypoints, sim.scenario.road_network)
        planner.plan(mission=mission)
        # 3. Apply agent vehicle association.
        vehicle = sim.vehicle_index.build_agent_vehicle(
            sim,
            agent_id,
            agent_interface,
            planner,
            sim.scenario.vehicle_filepath,
            sim.scenario.tire_parameters_filepath,
            True,
            sim.scenario.surface_patches,
            sim.scenario.controller_parameters_filepath,
            boid=False,
        )
        return vehicle

    def reset(self):
        self.captures_by_agent_id = defaultdict(list)

    def teardown(self):
        self.reset()
        self._traps.clear()

    def _mission2trap(self, road_network, mission, default_zone_dist=6):
        out_config = None
        if not (hasattr(mission, "start") and hasattr(mission, "goal")):
            raise ValueError(f"Value {mission} is not a mission!")

        activation_delay = mission.start_time
        patience = mission.entry_tactic.wait_to_hijack_limit_s
        zone = mission.entry_tactic.zone

        if not zone:
            n_lane = road_network.nearest_lane(mission.start.position)

            start_edge_id = n_lane.getEdge().getID()
            start_lane = n_lane.getIndex()
            lane_length = n_lane.getLength()
            lane_speed = n_lane.getSpeed()

            start_pos = mission.start.position
            vehicle_offset_into_lane = road_network.offset_into_lane(
                n_lane, (start_pos[0], start_pos[1])
            )
            vehicle_offset_into_lane = clip(
                vehicle_offset_into_lane, 1e-6, lane_length - 1e-6
            )

            drive_distance = lane_speed * default_zone_dist

            start_offset_in_lane = vehicle_offset_into_lane - drive_distance
            start_offset_in_lane = clip(start_offset_in_lane, 1e-6, lane_length - 1e-6)
            length = max(1e-6, vehicle_offset_into_lane - start_offset_in_lane)

            zone = MapZone(
                start=(start_edge_id, start_lane, start_offset_in_lane),
                length=length,
                n_lanes=1,
            )

        out_config = TrapConfig(
            zone=zone,
            # TODO: Make reactivation and activation delay configurable through
            #   scenario studio
            reactivation_time=-1,
            activation_delay=activation_delay,
            patience=patience,
            mission=mission,
            exclusion_prefixes=mission.entry_tactic.exclusion_prefixes,
        )

        return out_config
