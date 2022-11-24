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
from collections import Counter
from typing import Dict, List, Set

from .sensors import Sensor, Sensors, SensorState


class SensorManager:
    def __init__(self):
        self._sensors: Dict[str, Sensor] = {}

        # {vehicle_id (fixed-length): <SensorState>}
        self._sensor_states: Dict[str, SensorState] = {}

        self._sensors_by_actor_id: Dict[str, List[str]] = {}
        self._sensor_references = Counter()
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
        del self._sensor_states[actor_id]
        for sensor_id in self._sensors_by_actor_id[actor_id]:
            self._sensor_references.subtract([sensor_id])
            count = self._sensor_references[sensor_id]
            if count < 1:
                self._discarded_sensors.add(sensor_id)
        del self._sensors_by_actor_id[actor_id]

    def remove_sensor(self, sensor_id):
        sensor = self._sensors[sensor_id]
        del self._sensors[sensor_id]

    def sensor_state_exists(self, actor_id: str) -> bool:
        return actor_id in self._sensor_states

    def sensor_states_items(self):
        return self._sensor_states.items()

    def sensors_for_actor_id(self, actor_id):
        return [
            self._sensors[s_id] for s_id in self._sensors_by_actor_id.get(actor_id, [])
        ]

    def sensor_state_for_actor_id(self, actor_id):
        return self._sensor_states.get(actor_id)

    def add_sensor_for_actor(self, actor_id: str, sensor: Sensor):
        s_id = f"{type(sensor).__qualname__}-{id(actor_id)}"
        self._sensors[s_id] = sensor
        sensors = self._sensors_by_actor_id.setdefault(actor_id, [])
        sensors.append(s_id)
        self._sensor_references.update([s_id])

        return s_id

    def clean_up_sensors_for_actors(self, current_actor_ids: Set[str], renderer):
        """Cleans up sensors that are attached to non-existing actors."""
        # TODO MTA: Need to provide an explicit reference count of dependents on a sensor
        # TODO MTA: This is not good enough since actors can keep alive sensors that are not in use
        old_actor_ids = set(self._sensor_states)
        missing_actors = old_actor_ids - current_actor_ids

        for aid in missing_actors:
            self.remove_sensors_by_actor_id(aid)

        for sensor_id in self._discarded_sensors:
            if self._sensor_references.get(sensor_id) < 1:
                self._sensors[sensor_id].teardown(renderer=renderer)
        self._discarded_sensors.clear()
