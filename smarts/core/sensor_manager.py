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
import logging
from collections import Counter
from typing import Dict, List, Set

from .sensors import Sensor, Sensors, SensorState

logger = logging.getLogger(__name__)


class SensorManager:
    def __init__(self):
        self._sensors: Dict[str, Sensor] = {}

        # {actor_id: <SensorState>}
        self._sensor_states: Dict[str, SensorState] = {}

        # {actor_id: {sensor_id, ...}}
        self._sensors_by_actor_id: Dict[str, Set[str]] = {}
        self._actors_by_sensor_id: Dict[str, Set[str]] = {}
        self._sensor_references = Counter()
        # {sensor_id, ...}
        self._discarded_sensors: Set[str] = set()

    def step(self, sim_frame, renderer):
        for sensor_state in self._sensor_states.values():
            Sensors.step(sim_frame, sensor_state)

        for sensor in self._sensors.values():
            sensor.step(sim_frame=sim_frame, renderer=renderer)

    def teardown(self, renderer):
        for sensor in self._sensors.values():
            sensor.teardown(renderer=renderer)
        self._sensors = {}
        self._sensor_states = {}
        self._sensors_by_actor_id = {}
        self._sensor_references.clear()
        self._discarded_sensors.clear()

    def add_sensor_state(self, actor_id: str, sensor_state: SensorState):
        self._sensor_states[actor_id] = sensor_state

    def sensors(self):
        return self._sensors

    def remove_sensors_by_actor_id(self, actor_id: str):
        sensor_states = self._sensor_states.get(actor_id)
        if not sensor_states:
            logger.warning(
                "Attempted to remove sensors from actor with no sensors: `%s`",
                actor_id
            )
            return frozenset()
        del self._sensor_states[actor_id]
        sensors_by_actor = self._sensors_by_actor_id[actor_id]
        for sensor_id in sensors_by_actor:
            self._sensor_references.subtract([sensor_id])
            count = self._sensor_references[sensor_id]
            self._actors_by_sensor_id[sensor_id].remove(actor_id)
            if count < 1:
                self._discarded_sensors.add(sensor_id)
        del self._sensors_by_actor_id[actor_id]
        return frozenset(self._discarded_sensors)

    def remove_sensor(self, sensor_id):
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            return None
        del self._sensors[sensor_id]
        del self._sensor_references[sensor_id]

        ## clean up any remaining references by actors
        if sensor_id in self._actors_by_sensor_id:
            for actor_id in self._actors_by_sensor_id[sensor_id]:
                self._sensors_by_actor_id[actor_id].remove(sensor_id)
            del self._actors_by_sensor_id[sensor_id]
        return sensor

    def sensor_state_exists(self, actor_id: str) -> bool:
        return actor_id in self._sensor_states

    def sensor_states_items(self):
        return self._sensor_states.items()

    def sensors_for_actor_id(self, actor_id):
        return [
            self._sensors[s_id]
            for s_id in self._sensors_by_actor_id.get(actor_id, set())
        ]

    def sensor_state_for_actor_id(self, actor_id):
        return self._sensor_states.get(actor_id)

    def add_sensor_for_actor(self, actor_id: str, sensor: Sensor) -> str:
        # TAI: Allow multiple sensors of the same type on the same actor
        s_id = f"{type(sensor).__qualname__}-{actor_id}"
        actor_sensors = self._sensors_by_actor_id.setdefault(actor_id, set())
        if s_id in actor_sensors:
            logger.warning(
                "Duplicate sensor attempted to add to actor `%s`: `%s`", actor_id, s_id
            )
            return s_id
        actor_sensors.add(s_id)
        actors = self._actors_by_sensor_id.setdefault(s_id, set())
        actors.add(actor_id)
        return self.add_sensor(s_id, sensor)

    def add_sensor(self, sensor_id, sensor: Sensor) -> str:
        self._sensors[sensor_id] = sensor
        self._sensor_references.update([sensor_id])
        return sensor_id

    def clean_up_sensors_for_actors(self, current_actor_ids: Set[str], renderer):
        """Cleans up sensors that are attached to non-existing actors."""
        # This is not good enough by itself since actors can keep alive sensors that are not in use by an agent
        old_actor_ids = set(self._sensor_states)
        missing_actors = old_actor_ids - current_actor_ids

        for aid in missing_actors:
            self.remove_sensors_by_actor_id(aid)

        for sensor_id in self._discarded_sensors:
            if self._sensor_references.get(sensor_id) < 1:
                sensor = self.remove_sensor(sensor_id)
                sensor.teardown(renderer=renderer)

        self._discarded_sensors.clear()
