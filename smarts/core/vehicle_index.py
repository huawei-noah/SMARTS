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
from copy import copy, deepcopy
from io import StringIO
from typing import (
    FrozenSet,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import tableprint as tp

from smarts.core import gen_id
from smarts.core.utils import resources
from smarts.core.utils.cache import cache, clear_cache
from smarts.core.utils.string import truncate

from .actor import ActorRole
from .chassis import AckermannChassis, BoxChassis
from .controllers import ControllerState
from .road_map import RoadMap
from .sensors import SensorState
from .vehicle import Vehicle, VehicleState

VEHICLE_INDEX_ID_LENGTH = 128


def _2id(id_: str):
    separator = b"$"
    assert len(id_) <= VEHICLE_INDEX_ID_LENGTH - len(separator), id_

    if not isinstance(id_, bytes):
        id_ = bytes(id_.encode())

    return (separator + id_).zfill(VEHICLE_INDEX_ID_LENGTH - len(separator))


class _ControlEntity(NamedTuple):
    vehicle_id: Union[bytes, str]
    actor_id: Union[bytes, str]
    actor_role: ActorRole
    shadow_actor_id: Union[bytes, str]
    # Applies to shadowing and controlling actor
    # TODO: Consider moving this to an ActorRole field
    is_boid: bool
    is_hijacked: bool
    position: np.ndarray


# TODO: Consider wrapping the controlled_by recarry into a sep state object
#       VehicleIndex can perform operations on. Then we can do diffs of that
#       recarray with subset queries.
class VehicleIndex:
    """A vehicle management system that associates actors with vehicles."""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._controlled_by = VehicleIndex._build_empty_controlled_by()

        # Fixed-length ID to original ID
        # TODO: This quitely breaks if actor and vehicle IDs are the same. It assumes
        #       global uniqueness.
        self._2id_to_id = {}

        # {vehicle_id (fixed-length): <Vehicle>}
        self._vehicles = {}

        # {vehicle_id (fixed-length): <ControllerState>}
        self._controller_states = {}

        # {vehicle_id (fixed-length): <SensorState>}
        self._sensor_states = {}

        # Loaded from yaml file on scenario reset
        self._controller_params = {}

    @classmethod
    def identity(cls):
        """Returns an empty identity index."""
        return cls()

    def __sub__(self, other: "VehicleIndex") -> "VehicleIndex":
        vehicle_ids = set(self._controlled_by["vehicle_id"]) - set(
            other._controlled_by["vehicle_id"]
        )

        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return self._subset(vehicle_ids)

    def __and__(self, other: "VehicleIndex") -> "VehicleIndex":
        vehicle_ids = set(self._controlled_by["vehicle_id"]) & set(
            other._controlled_by["vehicle_id"]
        )

        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return self._subset(vehicle_ids)

    def _subset(self, vehicle_ids):
        assert self.vehicle_ids().issuperset(
            vehicle_ids
        ), f"{', '.join(list(self.vehicle_ids())[:3])} ⊅ {', '.join(list(vehicle_ids)[:3])}"

        vehicle_ids = [_2id(id_) for id_ in vehicle_ids]

        index = VehicleIndex()
        indices = np.isin(
            self._controlled_by["vehicle_id"], vehicle_ids, assume_unique=True
        )
        index._controlled_by = self._controlled_by[indices]
        index._2id_to_id = {id_: self._2id_to_id[id_] for id_ in vehicle_ids}
        index._vehicles = {id_: self._vehicles[id_] for id_ in vehicle_ids}
        index._controller_states = {
            id_: self._controller_states[id_]
            for id_ in vehicle_ids
            if id_ in self._controller_states
        }
        index._sensor_states = {
            id_: self._sensor_states[id_]
            for id_ in vehicle_ids
            if id_ in self._sensor_states
        }
        return index

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        dict_ = copy(self.__dict__)
        shallow = ["_2id_to_id", "_vehicles", "_sensor_states", "_controller_states"]
        for k in shallow:
            v = dict_.pop(k)
            setattr(result, k, copy(v))

        for k, v in dict_.items():
            setattr(result, k, deepcopy(v, memo))

        return result

    @cache
    def vehicle_ids(self) -> Set[str]:
        """A set of all unique vehicles ids in the index."""
        vehicle_ids = self._controlled_by["vehicle_id"]
        return {self._2id_to_id[id_] for id_ in vehicle_ids}

    @cache
    def agent_vehicle_ids(self) -> Set[str]:
        """A set of vehicle ids associated with an agent."""
        vehicle_ids = self._controlled_by[
            (self._controlled_by["actor_role"] == ActorRole.EgoAgent)
            | (self._controlled_by["actor_role"] == ActorRole.SocialAgent)
        ]["vehicle_id"]
        return {self._2id_to_id[id_] for id_ in vehicle_ids}

    @cache
    def social_vehicle_ids(
        self, vehicle_types: Optional[FrozenSet[str]] = None
    ) -> Set[str]:
        """A set of vehicle ids associated with traffic vehicles."""
        vehicle_ids = self._controlled_by[
            self._controlled_by["actor_role"] == ActorRole.Social
        ]["vehicle_id"]
        return {
            self._2id_to_id[id_]
            for id_ in vehicle_ids
            if not vehicle_types or self._vehicles[id_].vehicle_type in vehicle_types
        }

    @cache
    def vehicle_is_hijacked_or_shadowed(self, vehicle_id) -> Tuple[bool, bool]:
        """Determine if a vehicle is either taken over by an agent or watched by an agent."""
        vehicle_id = _2id(vehicle_id)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        if not np.any(v_index):
            return False, False

        vehicle = self._controlled_by[v_index]
        assert len(vehicle) == 1

        vehicle = vehicle[0]
        return bool(vehicle["is_hijacked"]), bool(vehicle["shadow_actor_id"])

    @cache
    def vehicle_ids_by_actor_id(self, actor_id, include_shadowers=False):
        """Returns all vehicles for the given actor ID as a list. This is most
        applicable when an agent is controlling multiple vehicles (e.g. with boids).
        """
        actor_id = _2id(actor_id)

        v_index = self._controlled_by["actor_id"] == actor_id
        if include_shadowers:
            v_index = v_index | (self._controlled_by["shadow_actor_id"] == actor_id)

        vehicle_ids = self._controlled_by[v_index]["vehicle_id"]
        return [self._2id_to_id[id_] for id_ in vehicle_ids]

    @cache
    def actor_id_from_vehicle_id(self, vehicle_id) -> Optional[str]:
        """Find the actor id associated with the given vehicle."""
        vehicle_id = _2id(vehicle_id)

        actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["actor_id"]

        if actor_ids:
            return self._2id_to_id[actor_ids[0]]

        return None

    @cache
    def shadow_actor_id_from_vehicle_id(self, vehicle_id) -> Optional[str]:
        """Find the first actor watching a vehicle."""
        vehicle_id = _2id(vehicle_id)

        shadow_actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["shadow_actor_id"]

        if shadow_actor_ids:
            return self._2id_to_id[shadow_actor_ids[0]]

        return None

    @cache
    def shadower_ids_from_vehicle_id(self, vehicle_id) -> Sequence[str]:
        """Find the first actor watching a vehicle."""
        vehicle_id = _2id(vehicle_id)

        shadow_actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["shadow_actor_id"]

        if shadow_actor_ids:
            return [
                self._2id_to_id[shadow_actor_id] for shadow_actor_id in shadow_actor_ids
            ]

        return []

    @cache
    def vehicle_position(self, vehicle_id):
        """Find the position of the given vehicle."""
        vehicle_id = _2id(vehicle_id)

        positions = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["position"]

        return positions[0] if len(positions) > 0 else None

    def vehicles_by_actor_id(self, actor_id, include_shadowers=False):
        """Find vehicles associated with the given actor.
        Args:
            actor_id:
                The actor to find all associated vehicle ids.
            include_shadowers:
                If to include vehicles that the actor is only watching.
        Returns:
            A list of associated vehicles.
        """
        vehicle_ids = self.vehicle_ids_by_actor_id(actor_id, include_shadowers)
        return [self._vehicles[_2id(id_)] for id_ in vehicle_ids]

    def vehicle_is_hijacked(self, vehicle_id: str) -> bool:
        """Determine if a vehicle is controlled by an actor."""
        is_hijacked, _ = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_hijacked

    def vehicle_is_shadowed(self, vehicle_id: str) -> bool:
        """Determine if a vehicle is watched by an actor."""
        _, is_shadowed = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_shadowed

    @property
    def vehicles(self):
        """A list of all existing vehicles."""
        return list(self._vehicles.values())

    @cache
    def vehicleitems(self) -> Iterator[Tuple[str, Vehicle]]:
        """A list of all vehicle IDs paired with their vehicle."""
        return map(lambda x: (self._2id_to_id[x[0]], x[1]), self._vehicles.items())

    @cache
    def vehicle_by_id(self, vehicle_id, default=...):
        """Get a vehicle by its id."""
        vehicle_id = _2id(vehicle_id)
        if default is ...:
            return self._vehicles[vehicle_id]
        return self._vehicles.get(vehicle_id, default)

    @clear_cache
    def teardown_vehicles_by_vehicle_ids(self, vehicle_ids):
        """Terminate and remove a vehicle from the index using its id."""
        self._log.debug(f"Tearing down vehicle ids: {vehicle_ids}")

        vehicle_ids = [_2id(id_) for id_ in vehicle_ids]
        if len(vehicle_ids) == 0:
            return

        for vehicle_id in vehicle_ids:
            vehicle = self._vehicles.pop(vehicle_id, None)
            if vehicle is not None:
                vehicle.teardown()

            # popping since sensor_states/controller_states may not include the
            # vehicle if it's not being controlled by an agent
            self._sensor_states.pop(vehicle_id, None)
            self._controller_states.pop(vehicle_id, None)

            # TODO: This stores actors/agents as well; those aren't being cleaned-up
            self._2id_to_id.pop(vehicle_id, None)

        remove_vehicle_indices = np.isin(
            self._controlled_by["vehicle_id"], vehicle_ids, assume_unique=True
        )

        self._controlled_by = self._controlled_by[~remove_vehicle_indices]

    def teardown_vehicles_by_actor_ids(self, actor_ids, include_shadowing=True):
        """Terminate and remove all vehicles associated with an actor."""
        vehicle_ids = []
        for actor_id in actor_ids:
            vehicle_ids.extend(
                [v.id for v in self.vehicles_by_actor_id(actor_id, include_shadowing)]
            )

        self.teardown_vehicles_by_vehicle_ids(vehicle_ids)

        return vehicle_ids

    @clear_cache
    def sync(self):
        """Update the state of the index."""
        for vehicle_id, vehicle in self._vehicles.items():
            v_index = self._controlled_by["vehicle_id"] == vehicle_id
            entity = _ControlEntity(*self._controlled_by[v_index][0])
            self._controlled_by[v_index] = tuple(
                entity._replace(position=vehicle.position)
            )

    @clear_cache
    def teardown(self):
        """Clean up resources, resetting the index."""
        self._controlled_by = VehicleIndex._build_empty_controlled_by()

        for vehicle in self._vehicles.values():
            vehicle.teardown(exclude_chassis=True)

        self._vehicles = {}
        self._controller_states = {}
        self._sensor_states = {}
        self._2id_to_id = {}

    @clear_cache
    def start_agent_observation(
        self, sim, vehicle_id, agent_id, agent_interface, plan, boid=False
    ):
        """Associate an agent to a vehicle. Set up any needed sensor requirements."""
        original_agent_id = agent_id
        vehicle_id, agent_id = _2id(vehicle_id), _2id(agent_id)

        vehicle = self._vehicles[vehicle_id]
        Vehicle.attach_sensors_to_vehicle(sim, vehicle, agent_interface, plan)

        self._2id_to_id[agent_id] = original_agent_id

        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps,
            plan=plan,
        )

        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = _ControlEntity(*self._controlled_by[v_index][0])
        self._controlled_by[v_index] = tuple(
            entity._replace(shadow_actor_id=agent_id, is_boid=boid)
        )

        # XXX: We are not giving the vehicle an AckermannChassis here but rather later
        #      when we switch_to_agent_control. This means when control that requires
        #      an AckermannChassis comes online, it needs to appropriately initialize
        #      chassis-specific states, such as tire rotation.

        return vehicle

    @clear_cache
    def switch_control_to_agent(
        self,
        sim,
        vehicle_id,
        agent_id,
        boid=False,
        hijacking=False,
        recreate=False,
        agent_interface=None,
    ):
        """Give control of the specified vehicle to the specified agent.
        Args:
            sim:
                An instance of a SMARTS simulation.
            vehicle_id:
                The id of the vehicle to associate.
            agent_id:
                The id of the agent to associate.
            boid:
                If the agent is acting as a boid agent controlling multiple vehicles.
            hijacking:
                If the vehicle has been taken over from another controlling actor.
            recreate:
                If the vehicle should be destroyed and regenerated.
            agent_interface:
                The agent interface for sensor requirements.
        """
        self._log.debug(f"Switching control of {agent_id} to {vehicle_id}")

        vehicle_id, agent_id = _2id(vehicle_id), _2id(agent_id)
        if recreate:
            # XXX: Recreate is presently broken for bubbles because it impacts the
            #      sumo traffic sim sync(...) logic in how it detects a vehicle as
            #      being hijacked vs joining. Presently it's still used for trapping.
            return self._switch_control_to_agent_recreate(
                sim, vehicle_id, agent_id, boid, hijacking
            )

        vehicle = self._vehicles[vehicle_id]
        chassis = None
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
            chassis = AckermannChassis(pose=vehicle.pose, bullet_client=sim.bc)
        else:
            chassis = BoxChassis(
                pose=vehicle.pose,
                speed=vehicle.speed,
                dimensions=vehicle.state.dimensions,
                bullet_client=sim.bc,
            )

        vehicle.swap_chassis(chassis)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = _ControlEntity(*self._controlled_by[v_index][0])
        actor_role = ActorRole.SocialAgent if hijacking else ActorRole.EgoAgent
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_role=actor_role,
                actor_id=agent_id,
                shadow_actor_id=b"",
                is_boid=boid,
                is_hijacked=hijacking,
            )
        )

        return vehicle

    @clear_cache
    def stop_shadowing(self, shadower_id: str, vehicle_id: Optional[str] = None):
        """Ends the shadowing by an a shadowing observer.

        Args:
            shadower_id (str): Removes this shadowing observer from all vehicles.
            vehicle_id (str, optional):
                If given this method removes shadowing from a specific vehicle. Defaults to None.
        """
        shadower_id = _2id(shadower_id)

        v_index = self._controlled_by["shadow_actor_id"] == shadower_id
        if vehicle_id:
            vehicle_id = _2id(vehicle_id)
            # This multiplication finds overlap of "shadow_actor_id" and "vehicle_id"
            v_index = (self._controlled_by["vehicle_id"] == vehicle_id) * v_index

        for entity in self._controlled_by[v_index]:
            entity = _ControlEntity(*entity)
            self._controlled_by[v_index] = tuple(entity._replace(shadow_actor_id=b""))

    @clear_cache
    def stop_agent_observation(self, vehicle_id):
        """Strip all sensors from a vehicle and stop all actors from watching the vehicle."""
        vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[vehicle_id]
        # pytype: disable=attribute-error
        Vehicle.detach_all_sensors_from_vehicle(vehicle)
        # pytype: enable=attribute-error

        # TAI: del self._sensor_states[vehicle_id]
        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = self._controlled_by[v_index][0]
        entity = _ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(entity._replace(shadow_actor_id=b""))

        return vehicle

    @clear_cache
    def relinquish_agent_control(
        self, sim, vehicle_id: str
    ) -> Tuple[VehicleState, RoadMap.Route]:
        """Give control of the vehicle back to its original controller."""
        self._log.debug(f"Relinquishing agent control v_id={vehicle_id}")

        v_id = _2id(vehicle_id)

        ss = self._sensor_states[v_id]
        route = ss.plan.route
        self.stop_agent_observation(vehicle_id)

        vehicle = self._vehicles[v_id]
        box_chassis = BoxChassis(
            pose=vehicle.chassis.pose,
            speed=vehicle.chassis.speed,
            dimensions=vehicle.chassis.dimensions,
            bullet_client=sim.bc,
        )
        vehicle.swap_chassis(box_chassis)

        v_index = self._controlled_by["vehicle_id"] == v_id
        entity = self._controlled_by[v_index][0]
        entity = _ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_role=ActorRole.Social,
                actor_id=b"",
                shadow_actor_id=b"",
                is_boid=False,
                is_hijacked=False,
            )
        )

        return vehicle.state, route

    @clear_cache
    def attach_sensors_to_vehicle(self, sim, vehicle_id, agent_interface, plan):
        """Attach sensors as per the agent interface requirements to the specified vehicle."""
        vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[vehicle_id]
        Vehicle.attach_sensors_to_vehicle(sim, vehicle, agent_interface, plan)
        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps,
            plan=plan,
        )
        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

    def _switch_control_to_agent_recreate(
        self, sim, vehicle_id, agent_id, boid, hijacking
    ):
        # XXX: vehicle_id and agent_id are already fixed-length as this is an internal
        #      method.
        agent_id = self._2id_to_id[agent_id]

        # TODO: There existed a SUMO connection error bug
        #       (https://gitlab.smartsai.xyz/smarts/SMARTS/-/issues/671) that occurred
        #       during lane changing when we hijacked/trapped a SUMO vehicle. Forcing
        #       vehicle recreation seems to address the problem. Ideally we discover
        #       the underlaying problem and can go back to the preferred implementation
        #       of simply swapping control of a persistent vehicle.
        # Get the old state values from the shadowed vehicle
        agent_interface = sim.agent_manager.agent_interface_for_agent_id(agent_id)
        assert (
            agent_interface is not None
        ), f"Missing agent_interface for agent_id={agent_id}"
        vehicle = self._vehicles[vehicle_id]
        sensor_state = self._sensor_states[vehicle_id]
        controller_state = self._controller_states[vehicle_id]
        plan = sensor_state.plan

        # Create a new vehicle to replace the old one
        new_vehicle = Vehicle.build_agent_vehicle(
            sim,
            vehicle.id,
            agent_interface,
            plan,
            sim.scenario.vehicle_filepath,
            sim.scenario.tire_parameters_filepath,
            not hijacking,
            sim.scenario.surface_patches,
        )

        # Apply the physical values from the old vehicle chassis to the new one
        new_vehicle.chassis.inherit_physical_values(vehicle.chassis)

        # Reserve space inside the traffic sim
        for traffic_sim in sim.traffic_sims:
            if traffic_sim.manages_actor(vehicle.id):
                traffic_sim.reserve_traffic_location_for_vehicle(
                    vehicle.id, vehicle.chassis.to_polygon
                )

        # Remove the old vehicle
        self.teardown_vehicles_by_vehicle_ids([vehicle.id])
        # HACK: Directly remove the vehicle from the traffic provider (should do this via the sim instead)
        for traffic_sim in sim.traffic_sims:
            if traffic_sim.manages_actor(vehicle.id):
                # TAI:  we probably should call "remove_vehicle(vehicle.id)" here instead,
                # and then call "add_vehicle(new_vehicle.state)", but since
                # the old and new vehicle-id and state are supposed to be the same
                # we take this short-cut.
                traffic_sim.stop_managing(vehicle.id)

        # Take control of the new vehicle
        self._enfranchise_actor(
            sim,
            agent_id,
            agent_interface,
            new_vehicle,
            controller_state,
            sensor_state,
            boid,
            hijacking,
        )

        return new_vehicle

    def build_agent_vehicle(
        self,
        sim,
        agent_id,
        agent_interface,
        plan,
        filepath,
        tire_filepath,
        trainable,
        surface_patches,
        initial_speed=None,
        boid=False,
    ):
        """Build an entirely new vehicle for an agent."""
        vehicle = Vehicle.build_agent_vehicle(
            sim=sim,
            vehicle_id=agent_id,
            agent_interface=agent_interface,
            plan=plan,
            vehicle_filepath=filepath,
            tire_filepath=tire_filepath,
            trainable=trainable,
            surface_patches=surface_patches,
            initial_speed=initial_speed,
        )

        sensor_state = SensorState(
            agent_interface.max_episode_steps,
            plan=plan,
        )

        controller_state = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

        self._enfranchise_actor(
            sim,
            agent_id,
            agent_interface,
            vehicle,
            controller_state,
            sensor_state,
            boid,
            hijacking=False,
        )

        return vehicle

    @clear_cache
    def _enfranchise_actor(
        self,
        sim,
        agent_id,
        agent_interface,
        vehicle,
        controller_state,
        sensor_state,
        boid: bool = False,
        hijacking: bool = False,
    ):
        # XXX: agent_id must be the original agent_id (not the fixed _2id(...))
        original_agent_id = agent_id

        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, sensor_state.plan
        )
        if sim.is_rendering:
            vehicle.create_renderer_node(sim.renderer)
            sim.renderer.begin_rendering_vehicle(vehicle.id, is_agent=True)

        vehicle_id = _2id(vehicle.id)
        agent_id = _2id(original_agent_id)

        self._sensor_states[vehicle_id] = sensor_state
        self._controller_states[vehicle_id] = controller_state
        self._vehicles[vehicle_id] = vehicle
        self._2id_to_id[vehicle_id] = vehicle.id
        self._2id_to_id[agent_id] = original_agent_id

        actor_role = ActorRole.SocialAgent if hijacking else ActorRole.EgoAgent
        entity = _ControlEntity(
            vehicle_id=vehicle_id,
            actor_id=agent_id,
            actor_role=actor_role,
            shadow_actor_id=b"",
            is_boid=boid,
            is_hijacked=hijacking,
            position=vehicle.position,
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

    @clear_cache
    def build_social_vehicle(
        self, sim, vehicle_state, actor_id, vehicle_id=None
    ) -> Vehicle:
        """Build an entirely new vehicle for a social agent."""
        if vehicle_id is None:
            vehicle_id = gen_id()

        vehicle = Vehicle.build_social_vehicle(
            sim,
            vehicle_id,
            vehicle_state,
        )

        vehicle_id, actor_id = _2id(vehicle_id), _2id(actor_id)
        if sim.is_rendering:
            vehicle.create_renderer_node(sim.renderer)
            sim.renderer.begin_rendering_vehicle(vehicle.id, is_agent=False)

        self._vehicles[vehicle_id] = vehicle
        self._2id_to_id[vehicle_id] = vehicle.id

        actor_role = vehicle_state.role
        assert actor_role not in (
            ActorRole.EgoAgent,
            ActorRole.SocialAgent,
        ), f"role={actor_role} from {vehicle_state.source}"
        entity = _ControlEntity(
            vehicle_id=vehicle_id,
            actor_id=actor_id,
            actor_role=actor_role,
            shadow_actor_id=b"",
            is_boid=False,
            is_hijacked=False,
            position=np.asarray(vehicle.position),
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

        return vehicle

    def begin_rendering_vehicles(self, renderer):
        """Render vehicles using the specified renderer."""
        agent_ids = self.agent_vehicle_ids()
        for vehicle in self._vehicles.values():
            if not vehicle.renderer:
                vehicle.create_renderer_node(renderer)
                is_agent = vehicle.id in agent_ids
                renderer.begin_rendering_vehicle(vehicle.id, is_agent)

    def sensor_states_items(self):
        """Get the sensor states of all listed vehicles."""
        return map(lambda x: (self._2id_to_id[x[0]], x[1]), self._sensor_states.items())

    def check_vehicle_id_has_sensor_state(self, vehicle_id: str) -> bool:
        """Determine if a vehicle contains sensors."""
        v_id = _2id(vehicle_id)
        return v_id in self._sensor_states

    def sensor_state_for_vehicle_id(self, vehicle_id: str) -> SensorState:
        """Retrieve the sensor state of the given vehicle."""
        vehicle_id = _2id(vehicle_id)
        return self._sensor_states[vehicle_id]

    @cache
    def controller_state_for_vehicle_id(self, vehicle_id: str) -> ControllerState:
        """Retrieve the controller state of the given vehicle."""
        vehicle_id = _2id(vehicle_id)
        return self._controller_states[vehicle_id]

    def load_controller_params(self, controller_filepath: str):
        """Set the default controller parameters for actor controlled vehicles."""
        self._controller_params = resources.load_controller_params(controller_filepath)

    def controller_params_for_vehicle_type(self, vehicle_type: str):
        """Get the controller parameters for the given vehicle type"""
        assert self._controller_params, "Controller params have not been loaded"
        return self._controller_params[vehicle_type]

    @staticmethod
    def _build_empty_controlled_by():
        return np.array(
            [],
            dtype=[
                # E.g. [(<vehicle ID>, <actor ID>, <actor type>), ...]
                ("vehicle_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("actor_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("actor_role", "B"),
                # XXX: Keeping things simple, this is always assumed to be an agent.
                #      We can add an shadow_actor_role when needed
                ("shadow_actor_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("is_boid", "B"),
                ("is_hijacked", "B"),
                ("position", np.float64, (3,)),
            ],
        )

    def __repr__(self):
        io = StringIO("")
        n_columns = len(self._controlled_by.dtype.names)

        by = self._controlled_by.copy().astype(
            list(zip(self._controlled_by.dtype.names, ["O"] * n_columns))
        )

        by["position"] = [", ".join([f"{x:.2f}" for x in p]) for p in by["position"]]
        by["actor_id"] = [truncate(p, 20) for p in by["actor_id"]]
        by["vehicle_id"] = [truncate(p, 20) for p in by["vehicle_id"]]
        by["shadow_actor_id"] = [truncate(p, 20) for p in by["shadow_actor_id"]]
        by["is_boid"] = [str(bool(x)) for x in by["is_boid"]]
        by["is_hijacked"] = [str(bool(x)) for x in by["is_hijacked"]]
        by["actor_role"] = [str(ActorRole(x)).split(".")[-1] for x in by["actor_role"]]

        # XXX: tableprint crashes when there's no data
        if by.size == 0:
            by = [[""] * n_columns]

        tp.table(by, self._controlled_by.dtype.names, style="round", out=io)
        return io.getvalue()
