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
import logging
from typing import Set

from smarts.core.sensors import SensorResolver, Sensors
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.simulation_local_constants import SimulationLocalConstants
from smarts.core.utils.file import replace
from smarts.core.utils.logging import timeit

logger = logging.getLogger(__name__)


class LocalSensorResolver(SensorResolver):
    """This implementation of the sensor resolver completes observations serially."""

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer,
        bullet_client,
    ):
        with timeit("serial run", logger.info):
            (
                observations,
                dones,
                updated_sensors,
            ) = Sensors.observe_serializable_sensor_batch(
                sim_frame,
                sim_local_constants,
                agent_ids,
            )

        # While observation processes are operating do rendering
        with timeit("rendering", logger.info):
            rendering = {}
            for agent_id in agent_ids:
                for vehicle_id in sim_frame.vehicles_for_agents[agent_id]:
                    (
                        rendering[agent_id],
                        updated_unsafe_sensors,
                    ) = Sensors.process_serialization_unsafe_sensors(
                        sim_frame,
                        sim_local_constants,
                        agent_id,
                        sim_frame.sensor_states[vehicle_id],
                        vehicle_id,
                        renderer,
                        bullet_client,
                    )
                    updated_sensors[vehicle_id].update(updated_unsafe_sensors)

        with timeit(f"merging observations", logger.info):
            # Merge sensor information
            for agent_id, r_obs in rendering.items():
                observations[agent_id] = replace(observations[agent_id], **r_obs)

        return observations, dones, updated_sensors

    def step(self, sim_frame, sensor_states):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()
