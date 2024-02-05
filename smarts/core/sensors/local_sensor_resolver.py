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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set, Tuple

from smarts.core.sensor import CustomRenderSensor
from smarts.core.sensors import SensorResolver, Sensors, SensorState
from smarts.core.utils.core_logging import timeit
from smarts.core.utils.file import replace

if TYPE_CHECKING:
    from smarts.core.observations import Observation
    from smarts.core.renderer_base import RendererBase
    from smarts.core.sensor import Sensor
    from smarts.core.simulation_frame import SimulationFrame
    from smarts.core.simulation_local_constants import SimulationLocalConstants
    from smarts.core.utils.pybullet import bullet_client as bc


logger = logging.getLogger(__name__)


class LocalSensorResolver(SensorResolver):
    """This implementation of the sensor resolver completes observations serially."""

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer: Optional[RendererBase],
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool], Dict[str, Dict[str, Sensor]]]:
        # If render buffer sensors:
        # Call serializable sensors
        # Call unserializable non-render results
        # Collect unserializable non-render results
        # Collect serializable results
        # Update renderable sensor buffers
        # Render
        # Collect renderable sensors
        # else:
        # Render
        # Call serializable sensors
        # Collect unserializable non-render sensors
        # Collect renderable sensors
        # Collect serializable sensors
        observations, dones, updated_sensors = self._gen_serialized_obs(
            sim_frame, sim_local_constants, agent_ids
        )
        phys_observations = self._gen_phys_observations(
            sim_frame, sim_local_constants, agent_ids, bullet_client, updated_sensors
        )

        # Merge physics sensor information
        for agent_id, p_obs in phys_observations.items():
            observations[agent_id] = replace(observations[agent_id], **p_obs)

        self._sync_custom_camera_sensors(sim_frame, renderer, observations)

        if renderer:
            renderer.render()

        rendering_observations = self._gen_rendered_observations(
            sim_frame, sim_local_constants, agent_ids, renderer, updated_sensors
        )

        # Merge sensor information
        for agent_id, r_obs in rendering_observations.items():
            observations[agent_id] = replace(observations[agent_id], **r_obs)

        return observations, dones, updated_sensors

    def _gen_serialized_obs(self, sim_frame, sim_local_constants, agent_ids):
        with timeit("serial run", logger.debug):
            (
                observations,
                dones,
                updated_sensors,
            ) = Sensors.observe_serializable_sensor_batch(
                sim_frame,
                sim_local_constants,
                agent_ids,
            )

        return observations, dones, updated_sensors

    def step(self, sim_frame: SimulationFrame, sensor_states: Iterable[SensorState]):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()
