# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set

from .controllers import ActionSpaceType, Controllers
from .provider import Provider, ProviderState
from .road_map import RoadMap
from .vehicle import ActorRole, VehicleState


class AgentsProvider(Provider):
    """Base class for providers that manage agent vehicles applying
    their actions via appropriate controllers to their chassis."""

    def __init__(self, sim):
        self._log = logging.getLogger(self.__class__.__name__)
        self._sim = sim
        self._my_agent_vehicles: Dict[str, List[VehicleState]] = dict()

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        # must be implemented by derived classes
        raise NotImplementedError

    def perform_agent_actions(self, agent_actions: Dict[str, Any]):
        """Applies any actions specified by agents controlling the vehicles managed by
        this provider via appropriate controllers to their chassis.
        Args:
            agent_actions: a dictionary from each agent_id to its actions for this step.
        """
        # must be implemented by derived classes
        raise NotImplementedError

    def setup(self, scenario) -> ProviderState:
        return self._provider_state

    def teardown(self):
        self._my_agent_vehicles = dict()

    def sync(self, provider_state):
        # Here we don't care what other providers are doing (pybullet cares for us).
        # The removal of missing vehicles is done in perform_agent_actions() instead of here
        # b/c these are agent vehicles and so get removed when they are done.
        pass

    def reset(self):
        self.teardown()

    def _remove_missing_vehicles(self, agent_actions: Dict[str, Any]):
        missing_agents = self._my_agent_vehicles.keys() - agent_actions.keys()
        for agent_id in missing_agents:
            self._log.info(
                f"agent {agent_id} no longer supplying actions, so provider will no longer manager its vehicle."
            )
            del self._my_agent_vehicles[agent_id]
        boid_vehicles_map = {
            vs.vehicle_id: agent_id
            for agent_id, vss in self._my_agent_vehicles.items()
            for vs in vss
            if len(vss) > 1
        }
        boid_vehicle_ids = boid_vehicles_map.keys()
        for agent_id, action in agent_actions.items():
            if self._sim.agent_manager.is_boid_agent(agent_id):
                boid_vehicle_ids -= action.keys()
        for missing in boid_vehicle_ids:
            self._log.info(
                f"boid agent {agent_id} no longer supplying actions for vehicle {missing}, so provider will no longer manage it."
            )
            boid_agent = boid_vehicles_map[missing]
            self._my_agent_vehicles[boid_agent] = [
                vs
                for vs in self._my_agent_vehicles[boid_agent]
                if vs.vehicle_id != missing
            ]

    def perform_agent_actions(self, agent_actions: Dict[str, Any]):
        """Apply actions to the vheicle(s) controlled by each managed agent
        via the appropriate controllers."""
        self._remove_missing_vehicles(agent_actions)
        actions_without_agents = agent_actions.keys() - self._my_agent_vehicles.keys()
        assert (
            not actions_without_agents
        ), f"actions specified for non-tracked agents:  {actions_without_agents}"
        for agent_id, vehicle_states in self._my_agent_vehicles.items():
            action = agent_actions.get(agent_id)
            if action is None or len(action) == 0:
                self._log.info(f"no actions for agent_id={agent_id}")
                continue
            agent_interface = self._sim.agent_manager.agent_interface_for_agent_id(
                agent_id
            )
            is_boid_agent = len(vehicle_states) > 1
            for vs in vehicle_states:
                vehicle = self._sim.vehicle_index.vehicle_by_id(vs.vehicle_id)
                vehicle_action = action[vehicle.id] if is_boid_agent else action
                controller_state = (
                    self._sim.vehicle_index.controller_state_for_vehicle_id(vehicle.id)
                )
                sensor_state = self._sim.vehicle_index.sensor_state_for_vehicle_id(
                    vehicle.id
                )
                Controllers.perform_action(
                    self._sim,
                    agent_id,
                    vehicle,
                    vehicle_action,
                    controller_state,
                    sensor_state,
                    agent_interface.action_space,
                    agent_interface.vehicle_type,
                )

    @property
    def _provider_state(self) -> ProviderState:
        provider_state = ProviderState()
        for agent_id, vehicle_states in self._my_agent_vehicles.items():
            for vs in vehicle_states:
                vehicle = self._sim.vehicle_index.vehicle_by_id(vs.vehicle_id)
                vs.pose = vehicle.pose
                vs.speed = vehicle.speed
                vs.source = self.source_str
                provider_state.vehicles.append(vs)
        return provider_state

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        # all actions should have already been handled by perform_agent_action()
        # TAI:  double-check that here?
        return self._provider_state

    def can_accept_vehicle(self, state: VehicleState) -> bool:
        return state.role == ActorRole.SocialAgent or state.role == ActorRole.EgoAgent

    def add_vehicle(
        self,
        provider_vehicle: VehicleState,
        route: Optional[Sequence[RoadMap.Route]] = None,
    ):
        provider_vehicle.source = self.source_str
        agent_id = self._sim.vehicle_index.actor_id_from_vehicle_id(
            provider_vehicle.vehicle_id
        )
        self._my_agent_vehicles.setdefault(agent_id, []).append(provider_vehicle)

    def _agent_for_vehicle(self, vehicle_id: str) -> Optional[str]:
        for agent_id, vss in self._my_agent_vehicles.items():
            for vs in vss:
                if vs.vehicle_id == vehicle_id:
                    return agent_id
        return None

    def manages_vehicle(self, vehicle_id: str) -> bool:
        return self._agent_for_vehicle(vehicle_id) is not None

    def stop_managing(self, vehicle_id: str):
        agent_id = self._agent_for_vehicle(vehicle_id)
        if agent_id:
            del self._my_agent_vehicles[agent_id]


class AgentPhysicsProvider(AgentsProvider):
    """A provider that manages agent vehicles with Ackermann chassis,
    applying their actions via appropriate controllers to their chassis."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {
            ActionSpaceType.Continuous,
            ActionSpaceType.Lane,
            ActionSpaceType.ActuatorDynamic,
            ActionSpaceType.LaneWithContinuousSpeed,
            ActionSpaceType.Trajectory,
            ActionSpaceType.MPC,
        }


class DirectControlProvider(AgentsProvider):
    """A provider that manages agent vehicles that are under direct control.
    These vehicles must have a Box chassis to allow for bypassing the application of forces."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.Direct}
