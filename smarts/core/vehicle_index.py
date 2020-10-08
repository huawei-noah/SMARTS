import logging
from io import StringIO
from enum import IntEnum
from collections import namedtuple

import numpy as np
import tableprint as tp

from smarts.core import gen_id
from .chassis import AckermannChassis, BoxChassis
from .vehicle import Vehicle, VehicleState
from .sensors import SensorState
from .controllers import ControllerState


class _ActorType(IntEnum):
    Social = 0
    Agent = 1


ControlEntity = namedtuple(
    "ControlEntity", ["vehicle_id", "actor_id", "actor_type", "shadow_actor_id",],
)


class VehicleIndex:
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        # XXX: Potentially the beginnings of "Multi-Index" data structure concept.
        self._controlled_by = self._build_empty_controlled_by()

        # {vehicle_id: <Vehicle>}
        self._vehicles = {}

        # {vehicle_id: <ControllerState>}
        self._controller_states = {}

        # {vehicle_id: <SensorState>}
        self._sensor_states = {}

    @property
    def agent_vehicle_ids(self):
        vehicle_ids = self._controlled_by[
            self._controlled_by["actor_type"] == _ActorType.Agent
        ]["vehicle_id"]
        return set(vehicle_ids)

    @property
    def social_vehicle_ids(self):
        vehicle_ids = self._controlled_by[
            self._controlled_by["actor_type"] == _ActorType.Social
        ]["vehicle_id"]
        return set(vehicle_ids)

    @property
    def vehicles(self):
        return list(self._vehicles.values())

    def vehicle_by_id(self, vehicle_id):
        return self._vehicles[vehicle_id]

    def teardown_vehicles_by_vehicle_ids(self, vehicle_ids):
        if not vehicle_ids:
            return

        self._log.debug(f"Tearing down vehicle ids: {vehicle_ids}")
        for vehicle_id in vehicle_ids:
            self._vehicles[vehicle_id].teardown()
            del self._vehicles[vehicle_id]

            # popping since sensor_states/controller_states may not include the
            # vehicle if it's not being controlled by an agent
            self._sensor_states.pop(vehicle_id, None)
            self._controller_states.pop(vehicle_id, None)

        remove_vehicle_indices = np.isin(
            self._controlled_by["vehicle_id"], list(vehicle_ids), assume_unique=True
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

    def teardown(self):
        self._controlled_by = self._build_empty_controlled_by()

        for vehicle in self._vehicles.values():
            vehicle.teardown(exclude_chassis=True)

        self._vehicles = {}
        self._controller_states = {}
        self._sensor_states = {}

    def vehicle_indices_by_actor_id(self, actor_id, columns):
        """Returns given vehicle index values for the given actor ID as list of columns.

        Args:
            actor_id:
                The id of the actor to query for.
            columns:
                The columns for the vehicles associated with the actor to retrieve.
        Returns:
            The columns of vehicle information related to the queried actor.
        """

        v_index = (self._controlled_by["actor_id"] == actor_id) | (
            self._controlled_by["shadow_actor_id"] == actor_id
        )
        return self._controlled_by[v_index][columns]

    def vehicle_ids_by_actor_id(self, actor_id, include_shadowers=False):
        """Returns all vehicles for the given actor ID as a list. This is most
        applicable when an agent is controlling multiple vehicles (e.g. with boids).
        """
        v_index = self._controlled_by["actor_id"] == actor_id
        if include_shadowers:
            v_index = v_index | (self._controlled_by["shadow_actor_id"] == actor_id)

        vehicle_ids = self._controlled_by[v_index]["vehicle_id"]
        return vehicle_ids

    def vehicles_by_actor_id(self, actor_id, include_shadowers=False):
        vehicle_ids = self.vehicle_ids_by_actor_id(actor_id, include_shadowers)
        return [self._vehicles[id_] for id_ in vehicle_ids]

    def actor_id_from_vehicle_id(self, vehicle_id):
        actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["actor_id"]

        return actor_ids[0] if actor_ids else None

    def shadow_actor_id_from_vehicle_id(self, vehicle_id):
        shadow_actor_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == vehicle_id
        ]["shadow_actor_id"]

        return shadow_actor_ids[0] if shadow_actor_ids else None

    def prepare_for_agent_control(
        self, sim, vehicle_id, agent_id, agent_interface, mission_planner
    ):
        vehicle = self._vehicles[vehicle_id]

        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, mission_planner
        )

        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )

        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.position, sim
        )

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = ControlEntity(*self._controlled_by[v_index][0])
        self._controlled_by[v_index] = tuple(entity._replace(shadow_actor_id=agent_id))

        # XXX: We are not giving the vehicle an AckermannChassis here but rather later
        #      when we switch_to_agent_control. This means when control that requires
        #      an AckermannChassis comes online, it needs to appropriately initialize
        #      chassis-specific states, such as tire rotation.

        return vehicle

    def attach_sensors_to_vehicle(
        self, sim, vehicle_id, agent_interface, mission_planner
    ):
        vehicle = self._vehicles[vehicle_id]

        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, mission_planner
        )
        self._sensor_states[vehicle_id] = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )
        self._controller_states[vehicle_id] = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.position, sim
        )

    def switch_control_to_agent(self, sim, vehicle_id, agent_id, recreate=False):
        self._log.debug(f"Switching control of {agent_id} to {vehicle_id}")
        if recreate:
            # XXX: Recreate is presently broken for bubbles because it impacts the
            #      sumo traffic sim sync(...) logic in how it detects a vehicle as
            #      being hijacked vs joining. Presently it's still used for trapping.
            return self._switch_control_to_agent_recreate(sim, vehicle_id, agent_id)

        vehicle = self._vehicles[vehicle_id]
        ackermann_chassis = AckermannChassis(pose=vehicle.pose, bullet_client=sim.bc)
        vehicle.swap_chassis(ackermann_chassis)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = ControlEntity(*self._controlled_by[v_index][0])
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_type=_ActorType.Agent, actor_id=agent_id, shadow_actor_id=""
            )
        )

        return vehicle

    def _switch_control_to_agent_recreate(self, sim, vehicle_id, agent_id):
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
        new_vehicle_id = vehicle_id
        new_vehicle = Vehicle.build_agent_vehicle(
            sim,
            new_vehicle_id,
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
            vehicle_id, vehicle.chassis.to_polygon
        )

        # Remove the old vehicle
        self.teardown_vehicles_by_vehicle_ids([vehicle_id])
        # HACK: Directly remove the vehicle from the traffic provider
        sim._traffic_sim.remove_traffic_vehicle(vehicle_id)

        # Take control of the new vehicle
        self._enfranchise_actor(
            sim,
            agent_id,
            agent_interface,
            new_vehicle_id,
            new_vehicle,
            controller_state,
            sensor_state,
        )

        return new_vehicle

    def relinquish_agent_control(self, sim, vehicle_id, social_vehicle_id):
        self._log.debug(
            f"Relinquishing agent control v_id={vehicle_id} sv_id={social_vehicle_id}"
        )
        vehicle = self._vehicles[vehicle_id]
        box_chassis = BoxChassis(
            pose=vehicle.chassis.pose,
            speed=vehicle.chassis.speed,
            dimensions=vehicle.chassis.dimensions,
            bullet_client=sim.bc,
        )
        vehicle.swap_chassis(box_chassis)
        Vehicle.detach_all_sensors_from_vehicle(vehicle)

        v_index = self._controlled_by["vehicle_id"] == vehicle_id
        entity = self._controlled_by[v_index][0]
        entity = ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(
            entity._replace(
                actor_type=_ActorType.Social,
                actor_id=social_vehicle_id,
                shadow_actor_id="",
            )
        )

        return vehicle

    # TODO: Collapse build_social_vehicle and build_agent_vehicle
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
    ):
        vehicle_id = f"{agent_id}-{str(gen_id())[:8]}"
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

        waypoint_paths = sim.waypoints.waypoint_paths_at(vehicle.position, lookahead=1)

        sensor_state = SensorState(
            agent_interface.max_episode_steps, mission_planner=mission_planner,
        )

        controller_state = ControllerState.from_action_space(
            agent_interface.action_space, vehicle.position, sim
        )

        self._enfranchise_actor(
            sim,
            agent_id,
            agent_interface,
            vehicle_id,
            vehicle,
            controller_state,
            sensor_state,
        )

        return vehicle

    def _enfranchise_actor(
        self,
        sim,
        agent_id,
        agent_interface,
        vehicle_id,
        vehicle,
        controller_state,
        sensor_state,
    ):
        Vehicle.attach_sensors_to_vehicle(
            sim, vehicle, agent_interface, sensor_state.mission_planner
        )
        vehicle.np.reparentTo(sim.vehicles_np)

        self._sensor_states[vehicle.id] = sensor_state
        self._controller_states[vehicle.id] = controller_state
        self._vehicles[vehicle.id] = vehicle
        self._controlled_by = np.insert(
            self._controlled_by, 0, (vehicle.id, agent_id, _ActorType.Agent, "")
        )

    def build_social_vehicle(
        self, sim, vehicle_state, actor_id, vehicle_type, vehicle_id=None
    ):
        if vehicle_id is None:
            vehicle_id = str(gen_id())

        vehicle = Vehicle.build_social_vehicle(
            sim, vehicle_id, vehicle_state, vehicle_type
        )
        vehicle.np.reparentTo(sim._root_np)

        self._vehicles[vehicle.id] = vehicle
        self._controlled_by = np.insert(
            self._controlled_by, 0, (vehicle.id, actor_id, _ActorType.Social, ""),
        )

        return vehicle

    def sensor_state_for_vehicle_id(self, vehicle_id):
        return self._sensor_states[vehicle_id]

    def controller_state_for_vehicle_id(self, vehicle_id):
        return self._controller_states[vehicle_id]

    def _build_empty_controlled_by(self):
        return np.array(
            [],
            dtype=[
                # E.g. [(<vehicle ID>, <actor ID>, <actor type>), ...]
                # TODO: Enforce fixed-length IDs in SMARTS so we can switch O to U.
                #       See https://numpy.org/doc/stable/reference/arrays.dtypes.html
                ("vehicle_id", "O"),
                ("actor_id", "O"),
                ("actor_type", "B"),
                # XXX: Keeping things simple, this is always assumed to be an agent.
                #      We can add an shadow_actor_type when needed
                ("shadow_actor_id", "O"),
            ],
        )

    def __repr__(self):
        io = StringIO("")
        table = tp.table(
            self._controlled_by, self._controlled_by.dtype.names, style="round", out=io,
        )
        return io.getvalue()
