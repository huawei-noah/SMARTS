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
from copy import copy, deepcopy
from enum import IntEnum
from io import StringIO
from typing import NamedTuple

import numpy as np
import tableprint as tp

from smarts.core import gen_id
from smarts.core.utils.string import truncate
from smarts.core.utils.cache import cache, clear_cache

from .chassis import AckermannChassis, BoxChassis
from .controllers import ControllerState
from .sensors import SensorState
from .vehicle import Vehicle

VEHICLE_INDEX_ID_LENGTH = 128


def _2id(id_: str):
    separator = b"$"
    assert len(id_) <= VEHICLE_INDEX_ID_LENGTH - len(separator), id_

    if not isinstance(id_, bytes):
        id_ = bytes(id_.encode())

    return (separator + id_).zfill(VEHICLE_INDEX_ID_LENGTH - len(separator))


class _ActorType(IntEnum):
    Social = 0
    Agent = 1


class _ControlEntity(NamedTuple):
    vehicle_id: str
    actor_id: str
    actor_type: _ActorType
    shadow_actor_id: str
    # Applies to shadowing and controlling actor
    # TODO: Consider moving this to an _ActorType field
    is_boid: bool
    is_hijacked: bool
    position: np.ndarray


# TODO: Consider wrapping the controlled_by recarry into a sep state object
#       VehicleIndex can perform operations on. Then we can do diffs of that
#       recarray with subset queries.
class VehicleIndex:
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

    @classmethod
    def identity(cls):
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
    def vehicle_ids(self):
        vehicle_ids = self._controlled_by["vehicle_id"]
        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return set(vehicle_ids)

    @cache
    def agent_vehicle_ids(self):
        vehicle_ids = self._controlled_by[
            self._controlled_by["actor_type"] == _ActorType.Agent
        ]["vehicle_id"]

        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return set(vehicle_ids)

    @cache
    def social_vehicle_ids(self):
        vehicle_ids = self._controlled_by[
            self._controlled_by["actor_type"] == _ActorType.Social
        ]["vehicle_id"]
        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return set(vehicle_ids)

    @cache
    def vehicle_is_hijacked_or_shadowed(self, vehicle_id):
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
    def actor_id_from_vehicle_id(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)

        actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["actor_id"]

        if actor_ids:
            return self._2id_to_id[actor_ids[0]]

        return None

    @cache
    def shadow_actor_id_from_vehicle_id(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)

        shadow_actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["shadow_actor_id"]

        if shadow_actor_ids:
            return self._2id_to_id[shadow_actor_ids[0]]

        return None

    @cache
    def vehicle_position(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)

        positions = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["position"]

        return positions[0] if len(positions) > 0 else None

    def vehicles_by_actor_id(self, actor_id, include_shadowers=False):
        vehicle_ids = self.vehicle_ids_by_actor_id(actor_id, include_shadowers)
        return [self._vehicles[_2id(id_)] for id_ in vehicle_ids]

    def vehicle_is_hijacked(self, vehicle_id):
        is_hijacked, _ = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_hijacked

    def vehicle_is_shadowed(self, vehicle_id):
        _, is_shadowed = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_shadowed

    @property
    def vehicles(self):
        # XXX: Order is not ensured
        return list(self._vehicles.values())

    def vehicleitems(self):
        return map(lambda x: (self._2id_to_id[x[0]], x[1]), self._vehicles.items())

    def vehicle_by_id(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)
        return self._vehicles[vehicle_id]

    @clear_cache
    def teardown_vehicles_by_vehicle_ids(self, vehicle_ids):
        self._log.debug(f"Tearing down vehicle ids: {vehicle_ids}")

        vehicle_ids = [_2id(id_) for id_ in vehicle_ids]
        if len(vehicle_ids) == 0:
            return

        for vehicle_id in vehicle_ids:
            self._vehicles[vehicle_id].teardown()
            del self._vehicles[vehicle_id]

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
        vehicle_ids = []
        for actor_id in actor_ids:
            vehicle_ids.extend(
                [v.id for v in self.vehicles_by_actor_id(actor_id, include_shadowing)]
            )

        self.teardown_vehicles_by_vehicle_ids(vehicle_ids)

        return vehicle_ids

    @clear_cache
    def sync(self):
        for vehicle_id, vehicle in self._vehicles.items():
            v_index = self._controlled_by["vehicle_id"] == vehicle_id
            entity = _ControlEntity(*self._controlled_by[v_index][0])
            self._controlled_by[v_index] = tuple(
                entity._replace(position=vehicle.position)
            )

    @clear_cache
    def teardown(self):
        self._controlled_by = VehicleIndex._build_empty_controlled_by()

        for vehicle in self._vehicles.values():
            vehicle.teardown(exclude_chassis=True)

        self._vehicles = {}
        self._controller_states = {}
        self._sensor_states = {}
        self._2id_to_id = {}

    @clear_cache
    def start_agent_observation(
        self, sim, vehicle_id, agent_id, agent_interface, mission_planner, boid=False
    ):
        original_agent_id = agent_id
        vehicle_id, agent_id = _2id(vehicle_id), _2id(agent_id)

        vehicle = self._vehicles[vehicle_id]
        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, mission_planner
        )

        self._2id_to_id[agent_id] = original_agent_id

        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )

        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.pose, sim
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
        self, sim, vehicle_id, agent_id, boid=False, hijacking=False, recreate=False
    ):
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
        ackermann_chassis = AckermannChassis(pose=vehicle.pose, bullet_client=sim.bc)
        vehicle.swap_chassis(ackermann_chassis)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = _ControlEntity(*self._controlled_by[v_index][0])
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_type=_ActorType.Agent,
                actor_id=agent_id,
                shadow_actor_id="",
                is_boid=boid,
                is_hijacked=hijacking,
            )
        )

        return vehicle

    @clear_cache
    def stop_agent_observation(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[vehicle_id]
        Vehicle.detach_all_sensors_from_vehicle(vehicle)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = self._controlled_by[v_index][0]
        entity = _ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(entity._replace(shadow_actor_id=""))

        return vehicle

    @clear_cache
    def relinquish_agent_control(self, sim, vehicle_id, social_vehicle_id):
        self._log.debug(
            f"Relinquishing agent control v_id={vehicle_id} sv_id={social_vehicle_id}"
        )

        vehicle_id, social_vehicle_id = _2id(vehicle_id), _2id(social_vehicle_id)

        vehicle = self._vehicles[vehicle_id]
        box_chassis = BoxChassis(
            pose=vehicle.chassis.pose,
            speed=vehicle.chassis.speed,
            dimensions=vehicle.chassis.dimensions,
            bullet_client=sim.bc,
        )
        vehicle.swap_chassis(box_chassis)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = self._controlled_by[v_index][0]
        entity = _ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_type=_ActorType.Social,
                actor_id="",
                shadow_actor_id="",
                is_boid=False,
                is_hijacked=False,
            )
        )

        return vehicle

    @clear_cache
    def attach_sensors_to_vehicle(
        self, sim, vehicle_id, agent_interface, mission_planner
    ):
        vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[vehicle_id]
        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, mission_planner
        )
        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )
        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.pose, sim
        )

    def _switch_control_to_agent_recreate(
        self, sim, vehicle_id, agent_id, boid, hijacking
    ):
        # XXX: vehicle_id and agent_id are already fixed-length as this is an internal
        #      method.
        agent_id = self._2id_to_id[agent_id]

        # TODO: There existed a SUMO connection error bug
        #       (https://gitlab.smartsai.xyz/smarts/SMARTS/-/issues/671) that occured
        #       during lange changing when we hijacked/trapped a SUMO vehicle. Forcing
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
        mission_planner = sensor_state.mission_planner

        # Create a new vehicle to replace the old one
        new_vehicle = Vehicle.build_agent_vehicle(
            sim,
            vehicle.id,
            agent_interface,
            mission_planner,
            sim.scenario.vehicle_filepath,
            sim.scenario.tire_parameters_filepath,
            # BUG: Both the TrapManager and BubbleManager call into this method but the
            #      trainable field below always assumes trainable=True
            True,
            sim.scenario.surface_patches,
            sim.scenario.controller_parameters_filepath,
        )

        # Apply the physical values from the old vehicle chassis to the new one
        new_vehicle.chassis.inherit_physical_values(vehicle.chassis)

        # Reserve space inside the traffic sim
        sim._traffic_sim.reserve_traffic_location_for_vehicle(
            vehicle.id, vehicle.chassis.to_polygon
        )

        # Remove the old vehicle
        self.teardown_vehicles_by_vehicle_ids([vehicle.id])
        # HACK: Directly remove the vehicle from the traffic provider
        sim._traffic_sim.remove_traffic_vehicle(vehicle.id)

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
        mission_planner,
        filepath,
        tire_filepath,
        trainable,
        surface_patches,
        controller_filepath,
        initial_speed=None,
        boid=False,
    ):
        vehicle_id = f"{agent_id}-{gen_id()}"
        vehicle = Vehicle.build_agent_vehicle(
            sim,
            vehicle_id,
            agent_interface,
            mission_planner,
            filepath,
            tire_filepath,
            trainable,
            surface_patches,
            controller_filepath,
            initial_speed,
        )

        sensor_state = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )

        controller_state = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.pose, sim
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
        boid=False,
        hijacking=False,
    ):
        # XXX: agent_id must be the original agent_id (not the fixed _2id(...))
        original_agent_id = agent_id

        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, sensor_state.mission_planner
        )
        vehicle.np.reparentTo(sim.vehicles_np)

        vehicle_id = _2id(vehicle.id)
        agent_id = _2id(original_agent_id)

        self._sensor_states[vehicle_id] = sensor_state
        self._controller_states[vehicle_id] = controller_state
        self._vehicles[vehicle_id] = vehicle
        self._2id_to_id[vehicle_id] = vehicle.id
        self._2id_to_id[agent_id] = original_agent_id

        entity = _ControlEntity(
            vehicle_id=vehicle_id,
            actor_id=agent_id,
            actor_type=_ActorType.Agent,
            shadow_actor_id="",
            is_boid=boid,
            is_hijacked=hijacking,
            position=vehicle.position,
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

    @clear_cache
    def build_social_vehicle(
        self, sim, vehicle_state, actor_id, vehicle_type, vehicle_id=None
    ):
        if vehicle_id is None:
            vehicle_id = gen_id()

        vehicle = Vehicle.build_social_vehicle(
            sim, vehicle_id, vehicle_state, vehicle_type
        )

        vehicle_id, actor_id = _2id(vehicle_id), _2id(actor_id)
        vehicle.np.reparentTo(sim._root_np)

        self._vehicles[vehicle_id] = vehicle
        self._2id_to_id[vehicle_id] = vehicle.id
        entity = _ControlEntity(
            vehicle_id=vehicle_id,
            actor_id=actor_id,
            actor_type=_ActorType.Social,
            shadow_actor_id="",
            is_boid=False,
            is_hijacked=False,
            position=vehicle.position,
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

        return vehicle

    def sensor_states_items(self):
        return map(lambda x: (self._2id_to_id[x[0]], x[1]), self._sensor_states.items())

    def sensor_state_for_vehicle_id(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)
        return self._sensor_states[vehicle_id]

    def controller_state_for_vehicle_id(self, vehicle_id):
        vehicle_id = _2id(vehicle_id)
        return self._controller_states[vehicle_id]

    @staticmethod
    def _build_empty_controlled_by():
        return np.array(
            [],
            dtype=[
                # E.g. [(<vehicle ID>, <actor ID>, <actor type>), ...]
                ("vehicle_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("actor_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("actor_type", "B"),
                # XXX: Keeping things simple, this is always assumed to be an agent.
                #      We can add an shadow_actor_type when needed
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
        by["actor_type"] = [str(_ActorType(x)).split(".")[-1] for x in by["actor_type"]]

        # XXX: tableprint crashes when there's no data
        if by.size == 0:
            by = [[""] * n_columns]

        tp.table(by, self._controlled_by.dtype.names, style="round", out=io)
        return io.getvalue()
