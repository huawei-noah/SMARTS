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
"""
.. spelling:word-list::

    shadower
    shadowers
"""
from __future__ import annotations

import logging
from copy import copy, deepcopy
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
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
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Dimensions, Pose
from smarts.core.utils import resources
from smarts.core.utils.cache import cache, clear_cache
from smarts.core.utils.strings import truncate
from smarts.core.vehicle_state import VEHICLE_CONFIGS

from .actor import ActorRole
from .chassis import AckermannChassis, BoxChassis
from .controllers import ControllerState
from .road_map import RoadMap
from .sensors import SensorState
from .vehicle import Vehicle

if TYPE_CHECKING:
    from smarts.core import plan
    from smarts.core.agent_interface import AgentInterface
    from smarts.core.controllers.action_space_type import ActionSpaceType
    from smarts.core.renderer_base import RendererBase
    from smarts.core.smarts import SMARTS

    from .vehicle import VehicleState

VEHICLE_INDEX_ID_LENGTH = 128


def _2id(id_: str) -> bytes:
    separator = b"$"
    assert len(id_) <= VEHICLE_INDEX_ID_LENGTH - len(separator), id_

    if not isinstance(id_, bytes):
        id_ = bytes(id_.encode())

    assert isinstance(id_, bytes)
    return (separator + id_).zfill(VEHICLE_INDEX_ID_LENGTH - len(separator))


class _ControlEntity(NamedTuple):
    vehicle_id: bytes
    owner_id: bytes
    role: ActorRole
    shadower_id: bytes
    # Applies to shadower and owner
    # TODO: Consider moving this to an ActorRole field
    is_boid: bool
    is_hijacked: bool
    position: np.ndarray


# TODO: Consider wrapping the controlled_by recarry into a sep state object
#       VehicleIndex can perform operations on. Then we can do diffs of that
#       recarray with subset queries.
class VehicleIndex:
    """A vehicle management system that associates owners with vehicles."""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._controlled_by = VehicleIndex._build_empty_controlled_by()

        # Fixed-length ID to original ID
        # TODO: This quitely breaks if owner and vehicle IDs are the same. It assumes
        #       global uniqueness.
        self._2id_to_id: Dict[bytes, str] = {}

        # {vehicle_id (fixed-length): <Vehicle>}
        self._vehicles: Dict[bytes, Vehicle] = {}

        # {vehicle_id (fixed-length): <ControllerState>}
        self._controller_states: Dict[bytes, ControllerState] = {}

        # Loaded from yaml file on scenario reset
        self._vehicle_definitions: resources.VehicleDefinitions = (
            resources.VehicleDefinitions({}, "")
        )

    @classmethod
    def identity(cls):
        """Returns an empty identity index."""
        return cls()

    def __sub__(self, other: VehicleIndex) -> VehicleIndex:
        vehicle_ids = set(self._controlled_by["vehicle_id"]) - set(
            other._controlled_by["vehicle_id"]
        )

        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return self._subset(vehicle_ids)

    def __and__(self, other: VehicleIndex) -> VehicleIndex:
        vehicle_ids = set(self._controlled_by["vehicle_id"]) & set(
            other._controlled_by["vehicle_id"]
        )

        vehicle_ids = [self._2id_to_id[id_] for id_ in vehicle_ids]
        return self._subset(vehicle_ids)

    def _subset(self, vehicle_ids: Iterable[str]):
        assert self.vehicle_ids().issuperset(
            vehicle_ids
        ), f"{', '.join(list(self.vehicle_ids())[:3])} ⊅ {', '.join(list(vehicle_ids)[:3])}"

        b_vehicle_ids = [_2id(id_) for id_ in vehicle_ids]

        index = VehicleIndex()
        indices = np.isin(
            self._controlled_by["vehicle_id"], b_vehicle_ids, assume_unique=True
        )
        index._controlled_by = self._controlled_by[indices]
        index._2id_to_id = {id_: self._2id_to_id[id_] for id_ in b_vehicle_ids}
        index._vehicles = {id_: self._vehicles[id_] for id_ in b_vehicle_ids}
        index._controller_states = {
            id_: self._controller_states[id_]
            for id_ in b_vehicle_ids
            if id_ in self._controller_states
        }
        return index

    def __deepcopy__(self, memo) -> VehicleIndex:
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        dict_ = copy(self.__dict__)
        shallow = ["_2id_to_id", "_vehicles", "_controller_states"]
        shallow += [c for c in dict_ if c.startswith("_cache_decorator_")]
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
            (self._controlled_by["role"] == ActorRole.EgoAgent)
            | (self._controlled_by["role"] == ActorRole.SocialAgent)
        ]["vehicle_id"]
        return {self._2id_to_id[id_] for id_ in vehicle_ids}

    @cache
    def social_vehicle_ids(
        self, vehicle_types: Optional[FrozenSet[str]] = None
    ) -> Set[str]:
        """A set of vehicle ids associated with traffic vehicles."""
        vehicle_ids = self._controlled_by[
            self._controlled_by["role"] == ActorRole.Social
        ]["vehicle_id"]
        return {
            self._2id_to_id[id_]
            for id_ in vehicle_ids
            if not vehicle_types or self._vehicles[id_].vehicle_type in vehicle_types
        }

    @cache
    def vehicle_is_hijacked_or_shadowed(self, vehicle_id: str) -> Tuple[bool, bool]:
        """Determine if a vehicle is either taken over by an owner or watched."""
        b_vehicle_id = _2id(vehicle_id)

        v_index = self._controlled_by["vehicle_id"] == b_vehicle_id
        if not np.any(v_index):
            return False, False

        vehicle = self._controlled_by[v_index]
        assert len(vehicle) == 1

        vehicle = vehicle[0]
        return bool(vehicle["is_hijacked"]), bool(vehicle["shadower_id"])

    @cache
    def vehicle_ids_by_owner_id(
        self, owner_id: str, include_shadowers: bool = False
    ) -> List[str]:
        """Returns all vehicles for the given owner ID as a list. This is most
        applicable when an agent is controlling multiple vehicles (e.g. with boids).
        """
        b_owner_id = _2id(owner_id)

        v_index = self._controlled_by["owner_id"] == b_owner_id
        if include_shadowers:
            v_index = v_index | (self._controlled_by["shadower_id"] == b_owner_id)

        b_vehicle_ids = self._controlled_by[v_index]["vehicle_id"]
        return [self._2id_to_id[id_] for id_ in b_vehicle_ids]

    @cache
    def owner_id_from_vehicle_id(self, vehicle_id: str) -> Optional[str]:
        """Find the owner id associated with the given vehicle."""
        b_vehicle_id = _2id(vehicle_id)

        owner_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == b_vehicle_id
        ]["owner_id"]

        if owner_ids:
            return self._2id_to_id[owner_ids[0]]

        return None

    @cache
    def shadower_id_from_vehicle_id(self, vehicle_id: str) -> Optional[str]:
        """Find the first shadowing entity watching a vehicle."""
        b_vehicle_id = _2id(vehicle_id)

        shadower_ids = self._controlled_by[
            self._controlled_by["vehicle_id"] == b_vehicle_id
        ]["shadower_id"]

        if shadower_ids:
            return self._2id_to_id[shadower_ids[0]]

        return None

    @cache
    def shadower_ids(self) -> Set[str]:
        """Get all current shadowing entity IDs."""
        return set(
            self._2id_to_id[sa_id]
            for sa_id in self._controlled_by["shadower_id"]
            if sa_id not in (b"", None)
        )

    @cache
    def vehicle_position(self, vehicle_id: str):
        """Find the position of the given vehicle."""
        b_vehicle_id = _2id(vehicle_id)

        positions = self._controlled_by[
            self._controlled_by["vehicle_id"] == b_vehicle_id
        ]["position"]

        return positions[0] if len(positions) > 0 else None

    def vehicles_by_owner_id(self, owner_id: str, include_shadowers: bool = False):
        """Find vehicles associated with the given owner id.
        Args:
            owner_id:
                The owner id to find all associated vehicle ids.
            include_shadowers:
                If to include vehicles that the owner is only watching.
        Returns:
            A list of associated vehicles.
        """
        vehicle_ids = self.vehicle_ids_by_owner_id(owner_id, include_shadowers)
        return [self._vehicles[_2id(id_)] for id_ in vehicle_ids]

    def vehicle_is_hijacked(self, vehicle_id: str) -> bool:
        """Determine if a vehicle is controlled by an owner."""
        is_hijacked, _ = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_hijacked

    def vehicle_is_shadowed(self, vehicle_id: str) -> bool:
        """Determine if a vehicle is watched by an owner."""
        _, is_shadowed = self.vehicle_is_hijacked_or_shadowed(vehicle_id)
        return is_shadowed

    @property
    def vehicles(self) -> List[Vehicle]:
        """A list of all existing vehicles."""
        return list(self._vehicles.values())

    @cache
    def vehicleitems(self) -> Iterator[Tuple[str, Vehicle]]:
        """A list of all vehicle IDs paired with their vehicle."""
        return map(lambda x: (self._2id_to_id[x[0]], x[1]), self._vehicles.items())

    @cache
    def vehicle_by_id(
        self,
        vehicle_id: str,
        default: Optional[Union[Vehicle, Ellipsis.__class__]] = ...,
    ):
        """Get a vehicle by its id."""
        b_vehicle_id = _2id(vehicle_id)
        if default is ...:
            return self._vehicles[b_vehicle_id]
        return self._vehicles.get(b_vehicle_id, default)

    @clear_cache
    def teardown_vehicles_by_vehicle_ids(
        self, vehicle_ids: Sequence[str], renderer: Optional[RendererBase]
    ):
        """Terminate and remove a vehicle from the index using its id."""
        self._log.debug("Tearing down vehicle ids: %s", vehicle_ids)

        b_vehicle_ids = [_2id(id_) for id_ in vehicle_ids]
        if len(b_vehicle_ids) == 0:
            return

        for b_vehicle_id in b_vehicle_ids:
            vehicle = self._vehicles.pop(b_vehicle_id, None)
            if vehicle is not None:
                vehicle.teardown(renderer=renderer)

            # popping since sensor_states/controller_states may not include the
            # vehicle if it's not being controlled by an agent
            self._controller_states.pop(b_vehicle_id, None)

            # TODO: This stores agents as well; those aren't being cleaned-up
            self._2id_to_id.pop(b_vehicle_id, None)

        remove_vehicle_indices = np.isin(
            self._controlled_by["vehicle_id"], b_vehicle_ids, assume_unique=True
        )

        self._controlled_by = self._controlled_by[~remove_vehicle_indices]

    def teardown_vehicles_by_owner_ids(
        self,
        owner_ids: Iterable[str],
        renderer: RendererBase,
        include_shadowing: bool = True,
    ) -> List[str]:
        """Terminate and remove all vehicles associated with an owner id."""
        vehicle_ids: List[str] = []
        for owner_id in owner_ids:
            vehicle_ids.extend(
                [v.id for v in self.vehicles_by_owner_id(owner_id, include_shadowing)]
            )

        self.teardown_vehicles_by_vehicle_ids(vehicle_ids, renderer)

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
    def teardown(self, renderer: RendererBase):
        """Clean up resources, resetting the index."""
        self._controlled_by = VehicleIndex._build_empty_controlled_by()

        for vehicle in self._vehicles.values():
            vehicle.teardown(renderer=renderer, exclude_chassis=True)

        self._vehicles = {}
        self._controller_states = {}
        self._2id_to_id = {}

    @clear_cache
    def start_agent_observation(
        self,
        sim: SMARTS,
        vehicle_id: str,
        agent_id: str,
        agent_interface: AgentInterface,
        plan: "plan.Plan",
        boid: bool = False,
        initialize_sensors: bool = True,
    ):
        """Associate an agent to a vehicle. Set up any needed sensor requirements."""
        b_vehicle_id, b_agent_id = _2id(vehicle_id), _2id(agent_id)
        self._2id_to_id[b_agent_id] = agent_id

        vehicle = self._vehicles[b_vehicle_id]

        sim.sensor_manager.add_sensor_state(
            vehicle.id,
            SensorState(
                agent_interface.max_episode_steps,
                plan_frame=plan.frame(),
            ),
        )
        if initialize_sensors:
            Vehicle.attach_sensors_to_vehicle(
                sim.sensor_manager, sim, vehicle, agent_interface
            )

        self._controller_states[b_vehicle_id] = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

        v_index = self._controlled_by["vehicle_id"] == b_vehicle_id
        entity = _ControlEntity(*self._controlled_by[v_index][0])
        self._controlled_by[v_index] = tuple(
            entity._replace(shadower_id=b_agent_id, is_boid=boid)
        )

        # XXX: We are not giving the vehicle a chassis here but rather later
        #      when we switch_to_agent_control. This means when control that requires
        #      a chassis comes online, it needs to appropriately initialize
        #      chassis-specific states, such as tire rotation.

        return vehicle

    @clear_cache
    def switch_control_to_agent(
        self,
        sim: SMARTS,
        vehicle_id: str,
        agent_id: str,
        boid: bool = False,
        hijacking: bool = False,
        recreate: bool = False,
        agent_interface: Optional[AgentInterface] = None,
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
                If the vehicle has been taken over from another controlling owner.
            recreate:
                If the vehicle should be destroyed and regenerated.
            agent_interface:
                The agent interface for sensor requirements.
        """
        self._log.debug("Switching control of '%s' to '%s'", vehicle_id, agent_id)

        b_vehicle_id, b_agent_id = _2id(vehicle_id), _2id(agent_id)
        if recreate:
            # XXX: Recreate is presently broken for bubbles because it impacts the
            #      sumo traffic sim sync(...) logic in how it detects a vehicle as
            #      being hijacked vs joining. Presently it's still used for trapping.
            return self._switch_control_to_agent_recreate(
                sim, b_vehicle_id, b_agent_id, boid, hijacking
            )

        vehicle = self._vehicles[b_vehicle_id]
        chassis = None
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
            vehicle_definition = self._vehicle_definitions.load_vehicle_definition(
                agent_interface.vehicle_class
            )
            chassis = AckermannChassis(
                pose=vehicle.pose,
                bullet_client=sim.bc,
                vehicle_dynamics_filepath=vehicle_definition.get("dynamics_model"),
                tire_parameters_filepath=vehicle_definition.get("tire_params"),
                friction_map=sim.scenario.surface_patches,
                controller_parameters=self._vehicle_definitions.controller_params_for_vehicle_class(
                    agent_interface.vehicle_class
                ),
                chassis_parameters=self._vehicle_definitions.chassis_params_for_vehicle_class(
                    agent_interface.vehicle_class
                ),
                initial_speed=vehicle.speed,
            )
        else:
            chassis = BoxChassis(
                pose=vehicle.pose,
                speed=vehicle.speed,
                dimensions=vehicle.state.dimensions,
                bullet_client=sim.bc,
            )
        vehicle.swap_chassis(chassis)

        v_index = self._controlled_by["vehicle_id"] == b_vehicle_id
        entity = _ControlEntity(*self._controlled_by[v_index][0])
        role = ActorRole.SocialAgent if hijacking else ActorRole.EgoAgent
        self._controlled_by[v_index] = tuple(
            entity._replace(
                role=role,
                owner_id=b_agent_id,
                shadower_id=b"",
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

        v_index = self._controlled_by["shadower_id"] == shadower_id
        if vehicle_id:
            b_vehicle_id = _2id(vehicle_id)
            # This multiplication finds overlap of "shadower_id" and "vehicle_id"
            v_index = (self._controlled_by["vehicle_id"] == b_vehicle_id) * v_index

        for entity in self._controlled_by[v_index]:
            entity = _ControlEntity(*entity)
            self._controlled_by[v_index] = tuple(entity._replace(shadower_id=b""))

    @clear_cache
    def stop_agent_observation(self, vehicle_id: str) -> Vehicle:
        """Strip all sensors from a vehicle and stop all owners from watching the vehicle."""
        b_vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[b_vehicle_id]

        v_index = self._controlled_by["vehicle_id"] == b_vehicle_id
        entity = self._controlled_by[v_index][0]
        entity = _ControlEntity(*entity)
        self._controlled_by[v_index] = tuple(entity._replace(shadower_id=b""))

        return vehicle

    @clear_cache
    def relinquish_agent_control(
        self, sim: SMARTS, vehicle_id: str, road_map: RoadMap
    ) -> Tuple[VehicleState, Optional[RoadMap.Route]]:
        """Give control of the vehicle back to its original controller."""
        self._log.debug(f"Relinquishing agent control v_id={vehicle_id}")

        v_id = _2id(vehicle_id)

        ss = sim.sensor_manager.sensor_state_for_actor_id(vehicle_id)
        route = ss.get_plan(road_map).route if ss else None
        vehicle = self.stop_agent_observation(vehicle_id)

        # pytype: disable=attribute-error
        Vehicle.detach_all_sensors_from_vehicle(vehicle)
        # pytype: enable=attribute-error
        sim.sensor_manager.remove_actor_sensors_by_actor_id(vehicle_id)
        sim.sensor_manager.remove_sensor_state_by_actor_id(vehicle_id)

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
                role=ActorRole.Social,
                owner_id=b"",
                shadower_id=b"",
                is_boid=False,
                is_hijacked=False,
            )
        )

        return vehicle.state, route

    @clear_cache
    def attach_sensors_to_vehicle(
        self,
        sim: SMARTS,
        vehicle_id: str,
        agent_interface: AgentInterface,
        plan: "plan.Plan",
    ):
        """Attach sensors as per the agent interface requirements to the specified vehicle."""
        b_vehicle_id = _2id(vehicle_id)

        vehicle = self._vehicles[b_vehicle_id]
        Vehicle.attach_sensors_to_vehicle(
            sim.sensor_manager,
            sim,
            vehicle,
            agent_interface,
        )
        sim.sensor_manager.add_sensor_state(
            vehicle.id,
            SensorState(
                agent_interface.max_episode_steps,
                plan_frame=plan.frame(),
            ),
        )
        self._controller_states[b_vehicle_id] = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

    def _switch_control_to_agent_recreate(
        self,
        sim: SMARTS,
        b_vehicle_id: bytes,
        b_agent_id: bytes,
        boid: bool,
        hijacking: bool,
    ):
        # XXX: vehicle_id and agent_id are already fixed-length as this is an internal
        #      method.
        agent_id = self._2id_to_id[b_agent_id]

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
        vehicle = self._vehicles[b_vehicle_id]
        sensor_state = sim.sensor_manager.sensor_state_for_actor_id(vehicle.id)
        assert sensor_state is not None
        controller_state = self._controller_states[b_vehicle_id]
        plan = sensor_state.get_plan(sim.road_map)

        vehicle_definition = self._vehicle_definitions.load_vehicle_definition(
            agent_interface.vehicle_class
        )
        # Create a new vehicle to replace the old one
        new_vehicle = VehicleIndex._build_agent_vehicle(
            sim,
            vehicle.id,
            agent_interface.action,
            vehicle_definition.get("type"),
            agent_interface.vehicle_class,
            plan.mission,
            vehicle_definition.get("dynamics_model"),
            vehicle_definition.get("tire_params"),
            vehicle_definition.get("visual_model"),
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
        self.teardown_vehicles_by_vehicle_ids([vehicle.id], sim.renderer_ref)
        sim.sensor_manager.remove_actor_sensors_by_actor_id(vehicle.id)
        # HACK: Directly remove the vehicle from the traffic provider (should do this via the sim instead)
        for traffic_sim in sim.traffic_sims:
            if traffic_sim.manages_actor(vehicle.id):
                # TAI:  we probably should call "remove_vehicle(vehicle.id)" here instead,
                # and then call "add_vehicle(new_vehicle.state)", but since
                # the old and new vehicle-id and state are supposed to be the same
                # we take this short-cut.
                traffic_sim.stop_managing(vehicle.id)

        # Take control of the new vehicle
        self._enfranchise_agent(
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

    @classmethod
    def _build_agent_vehicle(
        cls,
        sim: SMARTS,
        vehicle_id: str,
        action: Optional[ActionSpaceType],
        vehicle_type: str,
        vehicle_class: str,
        mission: plan.NavigationMission,
        vehicle_dynamics_filepath: Optional[str],
        tire_filepath: str,
        visual_model_filepath: str,
        trainable: bool,
        surface_patches: Sequence[Dict[str, Any]],
        initial_speed: Optional[float] = None,
    ) -> Vehicle:
        """Create a new vehicle and set up sensors and planning information as required by the
        ego agent.
        """
        chassis_dims = Vehicle.agent_vehicle_dims(mission, default=vehicle_type)

        start = mission.start
        if start.from_front_bumper:
            start_pose = Pose.from_front_bumper(
                front_bumper_position=np.array(start.position[:2]),
                heading=start.heading,
                length=chassis_dims.length,
            )
        else:
            start_pose = Pose.from_center(start.position, start.heading)

        vehicle_color = SceneColors.Agent if trainable else SceneColors.SocialAgent
        controller_parameters = (
            sim.vehicle_index._vehicle_definitions.controller_params_for_vehicle_class(
                vehicle_class
            )
        )
        chassis_parameters = (
            sim.vehicle_index._vehicle_definitions.chassis_params_for_vehicle_class(
                vehicle_class
            )
        )

        chassis = None
        if action in sim.dynamic_action_spaces:
            if mission.vehicle_spec:
                logger = logging.getLogger(cls.__name__)
                logger.warning(
                    "setting vehicle dimensions on a AckermannChassis not yet supported"
                )
            chassis = AckermannChassis(
                pose=start_pose,
                bullet_client=sim.bc,
                vehicle_dynamics_filepath=vehicle_dynamics_filepath,
                tire_parameters_filepath=tire_filepath,
                friction_map=surface_patches,
                controller_parameters=controller_parameters,
                chassis_parameters=chassis_parameters,
                initial_speed=initial_speed,
            )
        else:
            chassis = BoxChassis(
                pose=start_pose,
                speed=initial_speed,
                dimensions=chassis_dims,
                bullet_client=sim.bc,
            )

        vehicle = Vehicle(
            id=vehicle_id,
            chassis=chassis,
            color=vehicle_color,
            vehicle_config_type=vehicle_type,
            vehicle_class=vehicle_class,
            visual_model_filepath=visual_model_filepath,
        )

        return vehicle

    def build_agent_vehicle(
        self,
        sim: SMARTS,
        agent_id: str,
        agent_interface: AgentInterface,
        plan: "plan.Plan",
        trainable: bool,
        initial_speed: Optional[float] = None,
        boid: bool = False,
        *,
        vehicle_id: Optional[str] = None,
    ) -> Vehicle:
        """Build an entirely new vehicle for an agent."""
        vehicle_definition = self._vehicle_definitions.load_vehicle_definition(
            agent_interface.vehicle_class
        )
        vehicle = VehicleIndex._build_agent_vehicle(
            sim=sim,
            vehicle_id=vehicle_id or agent_id,
            action=agent_interface.action,
            vehicle_type=vehicle_definition.get("type"),
            vehicle_class=agent_interface.vehicle_class,
            mission=plan.mission,
            vehicle_dynamics_filepath=vehicle_definition.get("dynamics_model"),
            tire_filepath=vehicle_definition.get("tire_params"),
            visual_model_filepath=vehicle_definition.get("visual_model"),
            trainable=trainable,
            surface_patches=sim.scenario.surface_patches,
            initial_speed=initial_speed,
        )

        sensor_state = SensorState(agent_interface.max_episode_steps, plan.frame())

        controller_state = ControllerState.from_action_space(
            agent_interface.action, vehicle.pose, sim
        )

        self._enfranchise_agent(
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
    def _enfranchise_agent(
        self,
        sim: SMARTS,
        agent_id: str,
        agent_interface: AgentInterface,
        vehicle: Vehicle,
        controller_state: ControllerState,
        sensor_state: SensorState,
        boid: bool = False,
        hijacking: bool = False,
    ):
        # XXX: agent_id must be the original agent_id (not the fixed _2id(...))

        Vehicle.attach_sensors_to_vehicle(
            sim.sensor_manager, sim, vehicle, agent_interface
        )
        if sim.is_rendering:
            vehicle.create_renderer_node(sim.renderer_ref)
            sim.renderer.begin_rendering_vehicle(vehicle.id, is_agent=True)

        b_vehicle_id = _2id(vehicle.id)
        b_agent_id = _2id(agent_id)

        sim.sensor_manager.add_sensor_state(vehicle.id, sensor_state)
        self._controller_states[b_vehicle_id] = controller_state
        self._vehicles[b_vehicle_id] = vehicle
        self._2id_to_id[b_vehicle_id] = vehicle.id
        self._2id_to_id[b_agent_id] = agent_id

        role = ActorRole.SocialAgent if hijacking else ActorRole.EgoAgent
        entity = _ControlEntity(
            vehicle_id=b_vehicle_id,
            owner_id=b_agent_id,
            role=role,
            shadower_id=b"",
            is_boid=boid,
            is_hijacked=hijacking,
            position=vehicle.position,
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

    @staticmethod
    def _build_social_vehicle(
        sim: SMARTS, vehicle_id: str, vehicle_state: VehicleState
    ) -> Vehicle:
        """Create a new unassociated vehicle."""
        dims = Dimensions.copy_with_defaults(
            vehicle_state.dimensions,
            VEHICLE_CONFIGS[vehicle_state.vehicle_config_type].dimensions,
        )
        chassis = BoxChassis(
            pose=vehicle_state.pose,
            speed=vehicle_state.speed,
            dimensions=dims,
            bullet_client=sim.bc,
        )
        return Vehicle(
            id=vehicle_id,
            chassis=chassis,
            vehicle_config_type=vehicle_state.vehicle_config_type,
            visual_model_filepath=None,
        )

    @clear_cache
    def build_social_vehicle(
        self,
        sim: SMARTS,
        vehicle_state: VehicleState,
        owner_id: str,
        vehicle_id: Optional[str] = None,
    ) -> Vehicle:
        """Build an entirely new vehicle for a social agent."""
        if vehicle_id is None:
            vehicle_id = gen_id()

        vehicle = VehicleIndex._build_social_vehicle(
            sim,
            vehicle_id,
            vehicle_state,
        )

        b_vehicle_id, b_owner_id = _2id(vehicle_id), _2id(owner_id) if owner_id else b""
        if sim.is_rendering:
            vehicle.create_renderer_node(sim.renderer_ref)
            sim.renderer.begin_rendering_vehicle(vehicle.id, is_agent=False)

        self._vehicles[b_vehicle_id] = vehicle
        self._2id_to_id[b_vehicle_id] = vehicle.id

        role = vehicle_state.role
        assert role not in (
            ActorRole.EgoAgent,
            ActorRole.SocialAgent,
        ), f"role={role} from {vehicle_state.source}"
        entity = _ControlEntity(
            vehicle_id=b_vehicle_id,
            owner_id=b_owner_id,
            role=role,
            shadower_id=b"",
            is_boid=False,
            is_hijacked=False,
            position=np.asarray(vehicle.position),
        )
        self._controlled_by = np.insert(self._controlled_by, 0, tuple(entity))

        return vehicle

    def begin_rendering_vehicles(self, renderer: RendererBase):
        """Render vehicles using the specified renderer."""
        agent_vehicle_ids = self.agent_vehicle_ids()
        for vehicle in self._vehicles.values():
            if vehicle.create_renderer_node(renderer):
                is_agent = vehicle.id in agent_vehicle_ids
                renderer.begin_rendering_vehicle(vehicle.id, is_agent)

    @cache
    def controller_state_for_vehicle_id(self, vehicle_id: str) -> ControllerState:
        """Retrieve the controller state of the given vehicle."""
        return self._controller_states[_2id(vehicle_id)]

    def load_vehicle_definitions_list(self, vehicle_definitions_filepath: str):
        """Loads in a list of vehicle definitions."""
        self._vehicle_definitions = resources.load_vehicle_definitions_list(
            vehicle_definitions_filepath
        )

    @staticmethod
    def _build_empty_controlled_by():
        return np.array(
            [],
            dtype=[
                # E.g. [(<vehicle ID>, <owner ID>, <owner type>), ...]
                ("vehicle_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("owner_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("role", "B"),
                # XXX: Keeping things simple, this is currently assumed to be an agent.
                #      We can add a shadower_role when needed
                ("shadower_id", f"|S{VEHICLE_INDEX_ID_LENGTH}"),
                ("is_boid", "B"),
                ("is_hijacked", "B"),
                ("position", np.float64, (3,)),
            ],
        )

    def __repr__(self) -> str:
        io = StringIO("")
        n_columns = len(self._controlled_by.dtype.names)

        by = self._controlled_by.copy().astype(
            list(zip(self._controlled_by.dtype.names, ["O"] * n_columns))
        )

        by["position"] = [", ".join([f"{x:.2f}" for x in p]) for p in by["position"]]
        by["owner_id"] = [str(truncate(p, 20)) for p in by["owner_id"]]
        by["vehicle_id"] = [str(truncate(p, 20)) for p in by["vehicle_id"]]
        by["shadower_id"] = [str(truncate(p, 20)) for p in by["shadower_id"]]
        by["is_boid"] = [str(bool(x)) for x in by["is_boid"]]
        by["is_hijacked"] = [str(bool(x)) for x in by["is_hijacked"]]
        by["role"] = [str(ActorRole(x)).split(".")[-1] for x in by["role"]]

        # XXX: tableprint crashes when there's no data
        if by.size == 0:
            by = [[""] * n_columns]

        tp.table(by, self._controlled_by.dtype.names, style="round", out=io)
        return io.getvalue()
