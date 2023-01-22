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
import weakref
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .actor import ActorRole, ActorState
from .controllers import ActionSpaceType, Controllers
from .provider import Provider, ProviderManager, ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap
from .vehicle import VehicleState


class AgentsProvider(Provider):
    """Base class for providers that manage agent actors applying their
    actions via appropriate controllers (to their chassis if vehicles)."""

    def __init__(self, sim):
        self._log = logging.getLogger(self.__class__.__name__)
        self.set_manager(sim)
        # start with the default recovery flags...
        self._recovery_flags = super().recovery_flags
        self._my_agent_actors: Dict[str, List[ActorState]] = dict()

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    def set_manager(self, manager: ProviderManager):
        self._sim = weakref.ref(manager)

    @property
    def _vehicle_index(self):
        sim = self._sim()
        assert sim
        # pytype: disable=attribute-error
        # TAI: consider adding to ProviderManager interface
        return sim.vehicle_index
        # pytype: enable=attribute-error

    @property
    def _agent_manager(self):
        sim = self._sim()
        assert sim
        # pytype: disable=attribute-error
        # TAI: consider adding to ProviderManager interface
        return sim.agent_manager
        # pytype: enable=attribute-error

    @property
    def actions(self) -> Set[ActionSpaceType]:
        # must be implemented by derived classes
        raise NotImplementedError

    def setup(self, scenario) -> ProviderState:
        return self._provider_state

    def teardown(self):
        self._my_agent_actors = dict()

    def sync(self, provider_state):
        # Here we don't care what other providers are doing (pybullet cares for us).
        # The removal of missing actors is done in perform_agent_actions() instead of here
        # b/c these are agent actors and so get removed when they are done.
        pass

    def reset(self):
        self.teardown()

    def _remove_actors_without_actions(self, agent_actions: Dict[str, Any]):
        missing_agents = self._my_agent_actors.keys() - agent_actions.keys()
        for agent_id in missing_agents:
            self._log.info(
                f"agent {agent_id} no longer supplying actions, so provider will no longer manage its vehicle."
            )
            del self._my_agent_actors[agent_id]
        boid_actors_map = {
            vs.actor_id: agent_id
            for agent_id, vss in self._my_agent_actors.items()
            for vs in vss
            if len(vss) > 1
        }
        boid_vehicle_ids = boid_actors_map.keys()
        for agent_id, action in agent_actions.items():
            if self._agent_manager.is_boid_agent(agent_id):
                boid_vehicle_ids -= action.keys()
        for missing in boid_vehicle_ids:
            self._log.info(
                f"boid agent {agent_id} no longer supplying actions for vehicle {missing}, so provider will no longer manage it."
            )
            boid_agent = boid_actors_map[missing]
            self._my_agent_actors[boid_agent] = [
                vs for vs in self._my_agent_actors[boid_agent] if vs.actor_id != missing
            ]

    def perform_agent_actions(self, agent_actions: Dict[str, Any]):
        """Applies any actions specified by agents controlling the actors managed by
        this provider via appropriate controllers (to their chassis if vehicles).
        Args:
            agent_actions: a dictionary from each agent_id to its actions for this step.
        """
        self._remove_actors_without_actions(agent_actions)
        agents_without_actors = agent_actions.keys() - self._my_agent_actors.keys()
        if agents_without_actors:
            self._log.error(
                "actions specified for an agent without an actor: %s. Cleaning up social agents.",
                agents_without_actors,
            )
            orphaned_social_agents = (
                self._agent_manager.social_agent_ids & agents_without_actors
            )
            self._agent_manager.teardown_social_agents(orphaned_social_agents)

        agent_manager = self._agent_manager
        vehicle_index = self._vehicle_index
        sim = self._sim()
        assert sim
        for agent_id, vehicle_states in self._my_agent_actors.items():
            action = agent_actions.get(agent_id)
            if action is None or len(action) == 0:
                self._log.info("no actions for agent_id=%s", agent_id)
                continue
            if agent_id in agents_without_actors:
                continue
            agent_interface = agent_manager.agent_interface_for_agent_id(agent_id)
            is_boid_agent = agent_manager.is_boid_agent(agent_id)
            for vs in vehicle_states:
                vehicle = vehicle_index.vehicle_by_id(vs.actor_id)
                vehicle_action = action[vehicle.id] if is_boid_agent else action
                controller_state = vehicle_index.controller_state_for_vehicle_id(
                    vehicle.id
                )
                sensor_state = vehicle_index.sensor_state_for_vehicle_id(vehicle.id)
                Controllers.perform_action(
                    sim,
                    agent_id,
                    vehicle,
                    vehicle_action,
                    controller_state,
                    sensor_state,
                    agent_interface.action,
                    agent_interface.vehicle_type,
                )

    @property
    def _provider_state(self) -> ProviderState:
        provider_state = ProviderState()
        for agent_id, vehicle_states in self._my_agent_actors.items():
            for vs in vehicle_states:
                vehicle = self._vehicle_index.vehicle_by_id(vs.actor_id)
                vs.pose = vehicle.pose
                vs.speed = vehicle.speed
                vs.source = self.source_str
                provider_state.actors.append(vs)
        return provider_state

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        # all actions should have already been handled by perform_agent_action()
        # TAI:  double-check that here?
        return self._provider_state

    def can_accept_actor(self, state: ActorState) -> bool:
        # for now, we force our actors to be vehicles...
        return isinstance(state, VehicleState) and (
            state.role == ActorRole.SocialAgent or state.role == ActorRole.EgoAgent
        )

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional[Provider] = None
    ):
        provider_actor.source = self.source_str
        agent_id = self._vehicle_index.actor_id_from_vehicle_id(provider_actor.actor_id)
        self._my_agent_actors.setdefault(agent_id, []).append(provider_actor)

    def _agent_for_vehicle(self, vehicle_id: str) -> Optional[Tuple[str, int]]:
        for agent_id, vss in self._my_agent_actors.items():
            for i, vs in enumerate(vss):
                if vs.actor_id == vehicle_id:
                    return agent_id, i
        return None

    def manages_actor(self, actor_id: str) -> bool:
        return self._agent_for_vehicle(actor_id) is not None

    def stop_managing(self, actor_id: str):
        agent_tup = self._agent_for_vehicle(actor_id)
        if agent_tup:
            agent_id, i = agent_tup
            self._my_agent_actors[agent_id].pop(i)
            if not self._my_agent_actors[agent_id]:
                del self._my_agent_actors[agent_id]


class AgentPhysicsProvider(AgentsProvider):
    """A provider that manages agent vehicles with Ackermann chassis,
    applying their actions via appropriate controllers to their chassis."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return {
            ActionSpaceType.Continuous,
            ActionSpaceType.Lane,
            ActionSpaceType.ActuatorDynamic,
            ActionSpaceType.LaneWithContinuousSpeed,
            ActionSpaceType.Trajectory,
            ActionSpaceType.MPC,
            ActionSpaceType.Empty,
        }


class DirectControlProvider(AgentsProvider):
    """A provider that manages agent vehicles that are under direct control.
    These vehicles must have a Box chassis to allow for bypassing the application of forces."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.Direct}


class MotionPlannerProvider(AgentsProvider):
    """A provider that reshapes agent vehicle motion to follow a motion plan."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return {
            ActionSpaceType.TargetPose,
            ActionSpaceType.RelativeTargetPose,
            ActionSpaceType.MultiTargetPose,
        }


class TrajectoryInterpolationProvider(AgentsProvider):
    """A provider used to perform trajectory interpolation on agent actors that request trajectory following."""

    def __init__(self, sim):
        super().__init__(sim)

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.TrajectoryWithTime}
