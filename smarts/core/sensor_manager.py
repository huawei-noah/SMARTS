# MIT License
#
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
from __future__ import annotations

import logging
from collections import Counter
from typing import (
    TYPE_CHECKING,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from smarts.core import config
from smarts.core.observations import Observation
from smarts.core.sensors import SensorResolver, Sensors, SensorState
from smarts.core.sensors.local_sensor_resolver import LocalSensorResolver
from smarts.core.sensors.parallel_sensor_resolver import ParallelSensorResolver
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.simulation_local_constants import SimulationLocalConstants

if TYPE_CHECKING:
    from smarts.core.agent_interface import AgentInterface
    from smarts.core.renderer_base import RendererBase
    from smarts.core.sensor import Sensor
    from smarts.core.utils.pybullet import bullet_client as bc
    from smarts.core.vehicle import Vehicle


class SensorManager:
    """A sensor management system that associates actors with sensors."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._sensors: Dict[str, Sensor] = {}

        # {actor_id: <SensorState>}
        self._sensor_states: Dict[str, SensorState] = {}

        # {actor_id: {sensor_id, ...}}
        self._sensors_by_actor_id: Dict[str, Set[str]] = {}
        self._actors_by_sensor_id: Dict[str, Set[str]] = {}
        self._sensor_references = Counter()
        # {sensor_id, ...}
        self._scheduled_sensors: List[Tuple[str, Sensor]] = []
        observation_workers = config()(
            "core", "observation_workers", default=0, cast=int
        )
        parallel_resolver = ParallelSensorResolver
        if (backing := config()("core", "sensor_parallelization")) == "ray":
            try:
                import ray

                from smarts.ray.sensors.ray_sensor_resolver import RaySensorResolver

                parallel_resolver = RaySensorResolver
            except ImportError:
                pass
        elif backing == "mp":
            pass
        else:
            raise LookupError(
                f"SMARTS_CORE_SENSOR_PARALLELIZATION={backing} is not a valid option."
            )
        self._sensor_resolver: SensorResolver = (
            parallel_resolver() if observation_workers > 0 else LocalSensorResolver()
        )

    def step(self, sim_frame: SimulationFrame, renderer: RendererBase):
        """Update sensor values based on the new simulation state."""
        self._sensor_resolver.step(sim_frame, self._sensor_states.values())

        for sensor in self._sensors.values():
            sensor.step(sim_frame=sim_frame, renderer=renderer)

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer_ref: RendererBase,
        physics_ref: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool]]:
        """Runs observations and updates the sensor states.
        Args:
            sim_frame (SimulationFrame):
                The current state from the simulation.
            sim_local_constants (SimulationLocalConstants):
                The values that should stay the same for a simulation over a reset.
            agent_ids (Set[str]):
                The agent ids to process.
            renderer_ref (RendererBase):
                The renderer (if any) that should be used.
            physics_ref:
                The physics client.
        """
        observations, dones, updated_sensors = self._sensor_resolver.observe(
            sim_frame,
            sim_local_constants,
            agent_ids,
            renderer_ref,
            physics_ref,
        )
        for actor_id, sensors in updated_sensors.items():
            for sensor_name, sensor in sensors.items():
                self._sensors[
                    SensorManager._actor_and_sensor_name_to_sensor_id(
                        sensor_name, actor_id
                    )
                ] = sensor

        return observations, dones

    def observe_batch(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        interface: AgentInterface,
        sensor_states: Dict[str, SensorState],
        vehicles: Dict[str, Vehicle],
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool]]:
        """Operates all sensors on a batch of vehicles for a single agent."""
        # TODO: Replace this with a more efficient implementation that _actually_
        #       does batching
        assert sensor_states.keys() == vehicles.keys()

        observations, dones = {}, {}
        for vehicle_id, vehicle in vehicles.items():
            sensor_state = sensor_states[vehicle_id]
            (
                observations[vehicle_id],
                dones[vehicle_id],
                updated_sensors,
            ) = Sensors.observe_vehicle(
                sim_frame,
                sim_local_constants,
                interface,
                sensor_state,
                vehicle,
                renderer,
                bullet_client,
            )
            for sensor_name, sensor in updated_sensors.items():
                self._sensors[
                    SensorManager._actor_and_sensor_name_to_sensor_id(
                        sensor_name, vehicle_id
                    )
                ] = sensor

        return observations, dones

    def teardown(self, renderer: RendererBase):
        """Tear down the current sensors and clean up any internal resources."""
        self._logger.info("++ Sensors and sensor states reset. ++")
        for sensor in self._sensors.values():
            sensor.teardown(renderer=renderer)
        self._sensors = {}
        self._sensor_states = {}
        self._sensors_by_actor_id = {}
        self._sensor_references.clear()
        self._scheduled_sensors.clear()

    def add_sensor_state(self, actor_id: str, sensor_state: SensorState):
        """Add a sensor state associated with a given actor."""
        self._logger.debug("Sensor state added for actor '%s'.", actor_id)
        self._sensor_states[actor_id] = sensor_state

    def remove_sensor_state_by_actor_id(self, actor_id: str):
        """Add a sensor state associated with a given actor."""
        self._logger.debug("Sensor state removed for actor '%s'.", actor_id)
        return self._sensor_states.pop(actor_id, None)

    def remove_actor_sensors_by_actor_id(
        self, actor_id: str, schedule_teardown: bool = True
    ) -> Iterable[Tuple[Sensor, int]]:
        """Remove association of an actor to sensors. If the sensor is no longer associated an actor, the
        sensor is scheduled to be removed."""
        sensor_states = self._sensor_states.get(actor_id)
        if not sensor_states:
            self._logger.warning(
                "Attempted to remove sensors from actor with no sensors: '%s'.",
                actor_id,
            )
            return []
        self.remove_sensor_state_by_actor_id(actor_id)
        sensors_by_actor = self._sensors_by_actor_id.get(actor_id)
        if not sensors_by_actor:
            return []
        self._logger.debug("Target sensor removal for actor '%s'.", actor_id)
        discarded_sensors = []
        for sensor_id in sensors_by_actor:
            self._actors_by_sensor_id[sensor_id].remove(actor_id)
            self._sensor_references.subtract([sensor_id])
            references = self._sensor_references[sensor_id]
            discarded_sensors.append((self._sensors[sensor_id], references))
            if references < 1:
                self._disassociate_sensor(sensor_id, schedule_teardown)
        del self._sensors_by_actor_id[actor_id]
        return discarded_sensors

    def remove_sensor(
        self, sensor_id: str, schedule_teardown: bool = False
    ) -> Optional[Sensor]:
        """Remove a sensor by its id. Removes any associations it has with actors."""
        self._logger.debug("Target removal of sensor '%s'.", sensor_id)
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            return None
        self._disassociate_sensor(sensor_id, schedule_teardown)
        return sensor

    def _disassociate_sensor(self, sensor_id: str, schedule_teardown: bool):
        if schedule_teardown:
            self._scheduled_sensors.append((sensor_id, self._sensors[sensor_id]))

        self._logger.info("Sensor '%s' removed from manager.", sensor_id)
        del self._sensors[sensor_id]
        del self._sensor_references[sensor_id]

        ## clean up any remaining references by actors
        if sensor_id in self._actors_by_sensor_id:
            for actor_id in self._actors_by_sensor_id[sensor_id]:
                if sensors_ids := self._sensors_by_actor_id[actor_id]:
                    sensors_ids.remove(sensor_id)
            del self._actors_by_sensor_id[sensor_id]

    def sensor_state_exists(self, actor_id: str) -> bool:
        """Determines if a actor has a sensor state associated with it."""
        return actor_id in self._sensor_states

    def sensor_states_items(self) -> Iterator[Tuple[str, SensorState]]:
        """Gets all actor to sensor state associations."""
        return map(lambda x: x, self._sensor_states.items())

    def sensors_for_actor_id(self, actor_id: str) -> List[Sensor]:
        """Gets all sensors associated with the given actor."""
        return [
            self._sensors[s_id]
            for s_id in self._sensors_by_actor_id.get(actor_id, set())
        ]

    def sensors_for_actor_ids(
        self, actor_ids: Set[str]
    ) -> Dict[str, Dict[str, Sensor]]:
        """Gets all sensors for the given actors."""
        return {
            actor_id: {
                SensorManager._actor_sid_to_sname(s_id): self._sensors[s_id]
                for s_id in self._sensors_by_actor_id.get(actor_id, set())
            }
            for actor_id in actor_ids
        }

    def sensor_state_for_actor_id(self, actor_id: str) -> Optional[SensorState]:
        """Gets the sensor state for the given actor."""
        return self._sensor_states.get(actor_id)

    @staticmethod
    def _actor_sid_to_sname(sensor_id: str) -> str:
        return sensor_id.partition("-")[0]

    @staticmethod
    def _actor_and_sensor_name_to_sensor_id(sensor_name: str, actor_id: str) -> str:
        return f"{sensor_name}-{actor_id}"

    def add_sensor_for_actor(self, actor_id: str, name: str, sensor: Sensor) -> str:
        """Adds a sensor association for a specific actor."""
        # TAI: Allow multiple sensors of the same type on the same actor
        s_id = SensorManager._actor_and_sensor_name_to_sensor_id(name, actor_id)
        actor_sensors = self._sensors_by_actor_id.setdefault(actor_id, set())
        if s_id in actor_sensors:
            self._logger.warning(
                "Duplicate sensor attempted to add to actor `%s`: `%s`", actor_id, s_id
            )
            return s_id
        actor_sensors.add(s_id)
        actors = self._actors_by_sensor_id.setdefault(s_id, set())
        actors.add(actor_id)
        return self.add_sensor(s_id, sensor)

    def add_sensor(self, sensor_id, sensor: Sensor) -> str:
        """Adds a sensor to the sensor manager."""
        self._logger.info("Added sensor '%s' to sensor manager.", sensor_id)
        assert sensor_id not in self._sensors
        self._sensors[sensor_id] = sensor
        self._sensor_references.update([sensor_id])
        return sensor_id

    def clean_up_sensors_for_actors(
        self, current_actor_ids: Set[str], renderer: RendererBase
    ):
        """Cleans up sensors that are attached to non-existing actors."""
        # This is not good enough by itself since actors can keep alive sensors that are not in use by an agent
        old_actor_ids = set(self._sensor_states)
        missing_actors = old_actor_ids - current_actor_ids

        for aid in missing_actors:
            self.remove_actor_sensors_by_actor_id(aid)

        for sensor_id, sensor in self._scheduled_sensors:
            self._logger.info("Sensor '%s' destroyed.", sensor_id)
            sensor.teardown(renderer=renderer)

        self._scheduled_sensors.clear()
