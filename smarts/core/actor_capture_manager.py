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
from __future__ import annotations

import warnings
from collections import namedtuple
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional

import smarts
from smarts.core.plan import NavigationMission, Plan
from smarts.sstudio.sstypes import ConditionRequires

if TYPE_CHECKING:
    import smarts.core.scenario
    from smarts.core.actor import ActorState
    from smarts.core.smarts import SMARTS
    from smarts.core.vehicle import Vehicle


class ActorCaptureManager:
    """The base for managers that handle transition of control of actors."""

    def step(self, sim: SMARTS):
        """Step the manager. Assume modification of existence and control of the simulation actors.

        Args:
            sim (smarts.core.smarts.SMARTS): The smarts simulation instance.
        """
        raise NotImplementedError()

    def reset(self, scenario: smarts.core.scenario.Scenario, sim: SMARTS):
        """Reset this manager.

        :param scenario: The scenario to initialize from.
        :type scenario: smarts.core.scenario.Scenario
        :param sim: The simulation this is associated to.
        :type scenario: smarts.core.smarts.SMARTS
        """
        raise NotImplementedError()

    def teardown(self):
        """Clean up any unmanaged resources this manager uses (e.g. file handles.)"""
        raise NotImplementedError()

    @classmethod
    def _make_new_vehicle(
        cls, sim: SMARTS, agent_id, mission, initial_speed, social=False
    ) -> Optional[Vehicle]:
        if social:
            return cls.__make_new_social_vehicle(sim, agent_id, initial_speed)
        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        plan = Plan(sim.road_map, mission)
        # Apply agent vehicle association.
        vehicle = sim.vehicle_index.build_agent_vehicle(
            sim,
            agent_id,
            agent_interface,
            plan,
            True,
            initial_speed=initial_speed,
            boid=False,
        )
        if vehicle is not None:
            sim.agent_manager.remove_pending_agent_ids({agent_id})
            sim.create_vehicle_in_providers(vehicle, agent_id, True)
        return vehicle

    @staticmethod
    def __make_new_social_vehicle(sim, agent_id, initial_speed):
        social_agent_spec, social_agent_model = sim.scenario.social_agents[agent_id]

        social_agent_model = replace(social_agent_model, initial_speed=initial_speed)
        sim.agent_manager.add_and_emit_social_agent(
            agent_id,
            social_agent_spec,
            social_agent_model,
        )
        vehicles = sim.vehicle_index.vehicles_by_owner_id(agent_id)

        return vehicles[0] if len(vehicles) else None

    @staticmethod
    def _take_existing_vehicle(
        sim: SMARTS,
        vehicle_id: str,
        agent_id: str,
        mission: NavigationMission,
        social=False,
    ) -> Optional[Vehicle]:
        if social:
            # MTA: TODO implement this section of actor capture.
            warnings.warn(
                f"Unable to capture for {agent_id} because social agent id capture not yet implemented."
            )
            return None
        vehicle = sim.switch_control_to_agent(
            vehicle_id, agent_id, mission, recreate=True, is_hijacked=False
        )
        if vehicle is not None:
            sim.agent_manager.remove_pending_agent_ids({agent_id})
            sim.create_vehicle_in_providers(vehicle, agent_id, True)
        return vehicle

    @classmethod
    def _gen_all_condition_kwargs(
        cls,
        agent_id: str,
        mission: NavigationMission,
        sim: SMARTS,
        actor_state: ActorState,
        condition_requires: ConditionRequires,
    ):
        return {
            **cls._gen_mission_condition_kwargs(agent_id, mission, condition_requires),
            **cls._gen_simulation_condition_kwargs(sim, condition_requires),
            **cls._gen_actor_state_condition_args(
                sim.road_map, actor_state, condition_requires
            ),
        }

    @staticmethod
    def _gen_mission_condition_kwargs(
        agent_id: str,
        mission: Optional[NavigationMission],
        condition_requires: ConditionRequires,
    ) -> Dict[str, Any]:
        out_kwargs = dict()

        if (
            ConditionRequires.any_mission_state & condition_requires
        ) == ConditionRequires.none:
            return out_kwargs

        if condition_requires.agent_id in condition_requires:
            out_kwargs[ConditionRequires.agent_id.name] = agent_id
        if ConditionRequires.mission in condition_requires:
            out_kwargs[ConditionRequires.mission.name] = mission
        return out_kwargs

    @staticmethod
    def _gen_simulation_condition_kwargs(
        sim: SMARTS, condition_requires: ConditionRequires
    ) -> Dict[str, Any]:
        out_kwargs = dict()

        if (
            ConditionRequires.any_simulation_state & condition_requires
        ) == ConditionRequires.none:
            return out_kwargs

        if ConditionRequires.time in condition_requires:
            out_kwargs[ConditionRequires.time.name] = sim.elapsed_sim_time
        if ConditionRequires.actor_ids in condition_requires:
            out_kwargs[
                ConditionRequires.actor_ids.name
            ] = sim.vehicle_index.vehicle_ids()
        if ConditionRequires.road_map in condition_requires:
            out_kwargs[ConditionRequires.road_map.name] = sim.road_map
        if ConditionRequires.actor_states in condition_requires:
            out_kwargs[ConditionRequires.actor_states.name] = [
                v.state for v in sim.vehicle_index.vehicles
            ]
        if ConditionRequires.simulation in condition_requires:
            out_kwargs[ConditionRequires.simulation.name] = sim

        return out_kwargs

    @staticmethod
    def _gen_actor_state_condition_args(
        road_map,
        actor_state: Optional[ActorState],
        condition_requires: ConditionRequires,
    ) -> Dict[str, Any]:
        out_kwargs = dict()

        if (
            ConditionRequires.any_current_actor_state & condition_requires
        ) == ConditionRequires.none:
            return out_kwargs

        from smarts.core.road_map import RoadMap

        assert isinstance(road_map, RoadMap)

        if ConditionRequires.current_actor_state in condition_requires:
            out_kwargs[ConditionRequires.current_actor_state.name] = actor_state
        if ConditionRequires.current_actor_road_status in condition_requires:
            current_actor_road_status = namedtuple(
                "actor_road_status", ["road", "off_road"], defaults=[None, False]
            )
            if hasattr(actor_state, "pose"):
                road = road_map.road_with_point(actor_state.pose.point)
                current_actor_road_status.road = road
                current_actor_road_status.off_road = not road
            out_kwargs[
                ConditionRequires.current_actor_road_status.name
            ] = current_actor_road_status

        return out_kwargs
