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


from dataclasses import replace
from typing import Optional

from smarts.core.plan import Plan
from smarts.core.vehicle import Vehicle


class ActorCaptureManager:
    """The base for managers that handle transition of control of actors."""

    def step(self, sim):
        """Step the manager. Assume modification of existence and control of the simulation actors.

        Args:
            sim (SMARTS): The smarts simulation instance.
        """
        raise NotImplementedError()

    def reset(self, scenario, sim):
        """Reset this manager."""
        raise NotImplementedError()

    def teardown(self):
        """Clean up any unmanaged resources this manager uses (e.g. file handles.)"""
        raise NotImplementedError()

    @classmethod
    def _make_new_vehicle(
        cls, sim, agent_id, mission, initial_speed, social=False
    ) -> Optional[Vehicle]:
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)
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
            sim.scenario.vehicle_filepath,
            sim.scenario.tire_parameters_filepath,
            True,
            sim.scenario.surface_patches,
            initial_speed=initial_speed,
            boid=False,
        )
        if vehicle is not None:
            sim.agent_manager.remove_pending_agent_ids({agent_id})
            sim.create_vehicle_in_providers(vehicle, agent_id, True)
        return vehicle

    @staticmethod
    def __make_new_social_vehicle(sim, agent_id, initial_speed):
        from smarts.core.smarts import SMARTS

        sim: SMARTS = sim
        social_agent_spec, social_agent_model = sim.scenario.social_agents[agent_id]

        social_agent_model = replace(social_agent_model, initial_speed=initial_speed)
        sim.agent_manager.add_and_emit_social_agent(
            agent_id,
            social_agent_spec,
            social_agent_model,
        )
        vehicles = sim.vehicle_index.vehicles_by_actor_id(agent_id)

        return vehicles[0] if len(vehicles) else None

    @staticmethod
    def _take_existing_vehicle(
        sim, vehicle_id, agent_id, mission, social=False
    ) -> Optional[Vehicle]:
        from smarts.core.smarts import SMARTS

        assert isinstance(sim, SMARTS)
        if social:
            # Not supported
            return None
        vehicle = sim.switch_control_to_agent(
            vehicle_id, agent_id, mission, recreate=True, is_hijacked=False
        )
        if vehicle is not None:
            sim.agent_manager.remove_pending_agent_ids({agent_id})
            sim.create_vehicle_in_providers(vehicle, agent_id, True)
        return vehicle
