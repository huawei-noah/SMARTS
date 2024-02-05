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

import concurrent.futures
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set, Tuple

import ray

from smarts.core import config
from smarts.core.sensors import SensorResolver, Sensors
from smarts.core.serialization.default import dumps, loads
from smarts.core.utils.core_logging import timeit
from smarts.core.utils.file import replace

if TYPE_CHECKING:
    from smarts.core.configuration import Config
    from smarts.core.observations import Observation
    from smarts.core.renderer_base import RendererBase
    from smarts.core.sensor import Sensor
    from smarts.core.sensors import SensorState
    from smarts.core.simulation_frame import SimulationFrame
    from smarts.core.simulation_local_constants import SimulationLocalConstants
    from smarts.core.utils.pybullet import bullet_client as bc

logger = logging.getLogger(__name__)


class RaySensorResolver(SensorResolver):
    """A version of the sensor resolver that uses "ray" in its underlying implementation.

    Args:
        process_count_override (Optional[int]): An override for how many workers should be used.
    """

    def __init__(self, process_count_override: Optional[int] = None) -> None:
        conf: Config = config()
        self._num_observation_workers = (
            conf(
                "ray",
                "num_cpus",
                default=conf("core", "observation_workers", default=8, cast=int),
                cast=int,
            )
            if process_count_override == None
            else max(1, process_count_override)
        )
        if not ray.is_initialized():
            ray.init(
                num_cpus=self._num_observation_workers,
                num_gpus=conf("ray", "num_gpus", cast=int),
                log_to_driver=conf("ray", "log_to_driver", cast=bool),
            )
        self._sim_local_constants: SimulationLocalConstants = None
        self._current_workers = []

    def get_ray_worker_actors(self, count: int):
        """Get the current "ray" worker actors.

        Args:
            count (int): The number of workers to get.

        Returns:
            Any: The "ray" remote worker handles.
        """
        if len(self._current_workers) != count:
            # we need to cache because using options(name) is extremely slow
            self._current_workers = [
                RayProcessWorker.options(
                    name=f"sensor_worker_{i}", get_if_exists=True
                ).remote()
                for i in range(count)
            ]
        return self._current_workers

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool], Dict[str, Dict[str, Sensor]]]:
        observations, dones, updated_sensors = {}, {}, defaultdict(dict)

        ray_actors = self.get_ray_worker_actors(self._num_observation_workers)
        len_workers = len(ray_actors)

        tasks = self._gen_tasks_for_serializable_sensors(
            sim_frame, sim_local_constants, agent_ids, ray_actors, len_workers
        )
        phys_observations = self._gen_phys_observations(
            sim_frame, sim_local_constants, agent_ids, bullet_client, updated_sensors
        )

        # Collect futures
        with timeit("waiting for observations", logger.debug):
            for fut in concurrent.futures.as_completed(
                [task.future() for task in tasks]
            ):
                obs, ds, u_sens = fut.result()
                observations.update(obs)
                dones.update(ds)
                for v_id, values in u_sens.items():
                    updated_sensors[v_id].update(values)

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

    def _gen_tasks_for_serializable_sensors(
        self, sim_frame, sim_local_constants, agent_ids, ray_actors, len_workers
    ):
        tasks = []
        with timeit(
            f"setting up parallizable observations with {len(agent_ids)} and {len(ray_actors)}",
            logger.debug,
        ):
            # Update remote state (if necessary)
            remote_sim_frame = ray.put(dumps(sim_frame))
            if self._sim_local_constants is None or (
                self._sim_local_constants.road_map_hash
                != sim_local_constants.road_map_hash
            ):
                remote_sim_local_constants = ray.put(dumps(sim_local_constants))
                for a in ray_actors:
                    a.update_local_constants.remote(remote_sim_local_constants)

            # Start remote tasks
            agent_ids_for_grouping = list(agent_ids)
            agent_groups = [
                frozenset(agent_ids_for_grouping[i::len_workers])
                for i in range(len_workers)
            ]
            for i, agent_group in enumerate(agent_groups):
                if not agent_group:
                    break
                with timeit(f"submitting {len(agent_group)} agents", logger.debug):
                    tasks.append(
                        ray_actors[i].do_work.remote(
                            remote_sim_frame=remote_sim_frame, agent_ids=agent_group
                        )
                    )

        return tasks

    def step(self, sim_frame: SimulationFrame, sensor_states: Iterable[SensorState]):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()


@ray.remote
class RayProcessWorker:
    """A `ray` based process worker for parallel operation on sensors."""

    def __init__(self) -> None:
        self._simulation_local_constants: Optional[SimulationLocalConstants] = None

    def update_local_constants(self, sim_local_constants: SimulationLocalConstants):
        """Updates the process worker.

        Args:
            sim_local_constants (SimulationLocalConstants | None): The current simulation reset state.
        """
        self._simulation_local_constants = loads(sim_local_constants)

    def do_work(self, remote_sim_frame: SimulationFrame, agent_ids: Set[str]):
        """Run the sensors against the current simulation state.

        Args:
            remote_sim_frame (SimulationFrame): The current simulation state.
            agent_ids (set[str]): The agent ids to operate on.

        Returns:
            tuple[dict, dict, dict]: The updated sensor states: (observations, dones, updated_sensors)
        """
        sim_frame = loads(remote_sim_frame)
        return Sensors.observe_serializable_sensor_batch(
            sim_frame, self._simulation_local_constants, agent_ids
        )
