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
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

import psutil

import smarts.core.serialization.default as serializer
from smarts.core import config
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


class ParallelSensorResolver(SensorResolver):
    """This implementation of the sensor resolver completes observations in parallel."""

    def __init__(self, process_count_override: Optional[int] = None) -> None:
        super().__init__()
        self._logger: logging.Logger = logging.getLogger("Sensors")
        self._sim_local_constants: SimulationLocalConstants = None
        self._workers: List[SensorsWorker] = []
        self._process_count_override: Optional[int] = process_count_override

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer: RendererBase,
        bullet_client: bc.BulletClient,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool], Dict[str, Dict[str, Sensor]]]:
        """Runs observations in parallel where possible.
        Args:
            sim_frame (SimulationFrame):
                The current state from the simulation.
            sim_local_constants (SimulationLocalConstants):
                The values that should stay the same for a simulation over a reset.
            agent_ids ({str, ...}):
                The agent ids to process.
            renderer (Optional[Renderer]):
                The renderer (if any) that should be used.
            bullet_client:
                The physics client.
        """
        observations, dones, updated_sensors = {}, {}, defaultdict(dict)

        num_spare_cpus = max(0, psutil.cpu_count(logical=False) - 1)
        used_processes = (
            min(
                config()("core", "observation_workers", default=8, cast=int),
                num_spare_cpus,
            )
            if self._process_count_override == None
            else max(1, self._process_count_override)
        )

        used_workers = self._gen_workers_for_serializable_sensors(
            sim_frame, sim_local_constants, agent_ids, used_processes
        )
        phys_observations = self._gen_phys_observations(
            sim_frame, sim_local_constants, agent_ids, bullet_client, updated_sensors
        )

        # Collect futures
        with timeit("waiting for observations", logger.debug):
            if used_workers:
                while agent_ids != set(observations):
                    assert all(
                        w.running for w in used_workers
                    ), "A process worker crashed."
                    for result in mp.connection.wait(
                        [worker.connection for worker in used_workers], timeout=5
                    ):
                        # pytype: disable=attribute-error
                        obs, ds, u_sens = result.recv()
                        # pytype: enable=attribute-error
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

    def _gen_workers_for_serializable_sensors(
        self, sim_frame, sim_local_constants, agent_ids, used_processes
    ):
        workers: List[SensorsWorker] = self.get_workers(
            used_processes, sim_local_constants=sim_local_constants
        )
        used_workers: List[SensorsWorker] = []
        with timeit(
            f"setting up parallizable observations with {len(agent_ids)} and {len(workers)}",
            logger.debug,
        ):
            agent_ids_for_grouping = list(agent_ids)
            agent_groups = [
                agent_ids_for_grouping[i::used_processes] for i in range(used_processes)
            ]
            worker_args = WorkerKwargs(sim_frame=sim_frame)
            for i, agent_group in enumerate(agent_groups):
                if not agent_group:
                    break
                with timeit(f"submitting {len(agent_group)} agents", logger.debug):
                    workers[i].send(
                        SensorsWorker.Request(
                            SensorsWorkerRequestId.SIMULATION_FRAME,
                            worker_args.merged(WorkerKwargs(agent_ids=agent_group)),
                        )
                    )
                    used_workers.append(workers[i])
        return used_workers

    def __del__(self):
        try:
            self.stop_all_workers()
        except AttributeError:
            pass

    def stop_all_workers(self):
        """Stop all current workers and clear reference to them."""
        for worker in self._workers:
            worker.stop()
        self._workers = []

    def _validate_configuration(self, local_constants: SimulationLocalConstants):
        """Check that constants have not changed which might indicate that the workers need to be updated."""
        return local_constants == self._sim_local_constants

    def generate_workers(
        self, count: int, workers_list: List[SensorsWorker], worker_kwargs: WorkerKwargs
    ):
        """Generate the given number of workers requested."""
        while len(workers_list) < count:
            new_worker = SensorsWorker()
            workers_list.append(new_worker)
            new_worker.run()
            new_worker.send(
                request=SensorsWorker.Request(
                    SensorsWorkerRequestId.SIMULATION_LOCAL_CONSTANTS, worker_kwargs
                )
            )

    def get_workers(
        self, count: int, sim_local_constants: SimulationLocalConstants, **kwargs
    ) -> List["SensorsWorker"]:
        """Get the give number of workers."""
        if not self._validate_configuration(sim_local_constants):
            self.stop_all_workers()
            self._sim_local_constants = sim_local_constants
        if len(self._workers) < count:
            worker_kwargs = WorkerKwargs(
                **kwargs, sim_local_constants=sim_local_constants
            )
            self.generate_workers(count, self._workers, worker_kwargs)
        return self._workers[:count]

    def step(self, sim_frame: SimulationFrame, sensor_states: Iterable[SensorState]):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()

    @property
    def process_count_override(self) -> Optional[int]:
        """The number of processes this implementation should run.

        Returns:
            int: Number of processes.
        """
        return self._process_count_override

    @process_count_override.setter
    def process_count_override(self, count: Optional[int]):
        self._process_count_override = count


class WorkerKwargs:
    """Used to serialize arguments for a worker upfront."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = self._serialize(kwargs)

    def merged(self, o_worker_kwargs: "WorkerKwargs") -> "WorkerKwargs":
        """Merge two worker arguments and return a new copy."""
        new = type(self)()
        new.kwargs = {**self.kwargs, **o_worker_kwargs.kwargs}
        return new

    @staticmethod
    def _serialize(kwargs: Dict):
        return {
            k: serializer.dumps(a) if a is not None else a for k, a in kwargs.items()
        }

    def deserialize(self):
        """Deserialize all objects in the arguments and return a dictionary copy."""
        return {
            k: serializer.loads(a) if a is not None else a
            for k, a in self.kwargs.items()
        }


class ProcessWorker:
    """A utility class that defines a persistent worker which will continue to operate in the background."""

    class WorkerDone:
        """The done signal for a worker."""

        pass

    @dataclass
    class Request:
        """A request to made to the process worker"""

        id: Any
        data: WorkerKwargs

    def __init__(self, serialize_results=False) -> None:
        parent_connection, child_connection = mp.Pipe()
        self._parent_connection = parent_connection
        self._child_connection = child_connection
        self._serialize_results = serialize_results
        self._proc: Optional[mp.Process] = None

    @classmethod
    def _do_work(cls, state):
        raise NotImplementedError()

    @classmethod
    def _on_request(cls, state: Dict, request: Request) -> bool:
        """
        Args:
            state: The persistent state on the worker
            request: A request made to the worker.

        Returns:
            bool: If the worker method `_do_work` should be called.
        """
        raise NotImplementedError()

    @classmethod
    def _run(
        cls: "ProcessWorker",
        connection: mp.connection.Connection,
        serialize_results,
    ):
        state: Dict[Any, Any] = {}
        while True:
            run_work = False
            work = connection.recv()
            if isinstance(work, cls.WorkerDone):
                break
            if isinstance(work, cls.Request):
                run_work = cls._on_request(state, request=work)
            with timeit("do work", logger.debug):
                if not run_work:
                    continue
                result = cls._do_work(state=state.copy())
                with timeit("reserialize", logger.debug):
                    if serialize_results:
                        result = serializer.dumps(result)
                with timeit("put back to main thread", logger.debug):
                    connection.send(result)

    def run(self):
        """Start the worker seeded with the given data."""
        kwargs = dict(serialize_results=self._serialize_results)
        # pytype: disable=wrong-arg-types
        self._proc = mp.Process(
            target=self._run,
            args=(self._child_connection,),
            kwargs=kwargs,
            daemon=True,
        )
        # pytype: enable=wrong-arg-types
        self._proc.start()
        return self._parent_connection

    def send(self, request: Request):
        """Sends a request to the worker."""
        assert isinstance(request, self.Request)
        self._parent_connection.send(request)

    def result(self, timeout=None):
        """The most recent result from the worker."""
        with timeit("main thread blocked", logger.debug):
            conn = mp.connection.wait([self._parent_connection], timeout=timeout).pop()
            # pytype: disable=attribute-error
            result = conn.recv()
            # pytype: enable=attribute-error
        with timeit("deserialize for main thread", logger.debug):
            if self._serialize_results:
                result = serializer.loads(result)
        return result

    def stop(self):
        """Sends a stop signal to the worker."""
        try:
            self._parent_connection.send(self.WorkerDone())
        except ImportError:
            # Python is shutting down.
            if not self._parent_connection.closed:
                self._parent_connection.close()

    @property
    def running(self) -> bool:
        """If this current worker is still running."""
        return self._proc is not None and self._proc.exitcode is None

    @property
    def connection(self):
        """The underlying connection to send data to the worker."""
        return self._parent_connection


class SensorsWorkerRequestId(IntEnum):
    """Options for update requests to a sensor worker."""

    SIMULATION_FRAME = 1
    SIMULATION_LOCAL_CONSTANTS = 2


class SensorsWorker(ProcessWorker):
    """A worker for sensors."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _do_work(cls, state):
        return cls.local(state=state)

    @classmethod
    def _on_request(cls, state: Dict, request: ProcessWorker.Request) -> bool:
        assert request.data is None or isinstance(request.data, WorkerKwargs)
        if request.id == SensorsWorkerRequestId.SIMULATION_FRAME:
            state.update(request.data.deserialize())
            return True
        if request.id == SensorsWorkerRequestId.SIMULATION_LOCAL_CONSTANTS:
            state.update(request.data.deserialize())

        return False

    @staticmethod
    def local(state: Dict):
        """The work method on the local thread."""
        sim_local_constants = state["sim_local_constants"]
        sim_frame = state["sim_frame"]
        agent_ids = state["agent_ids"]
        return Sensors.observe_serializable_sensor_batch(
            sim_frame, sim_local_constants, agent_ids
        )
