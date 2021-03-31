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
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pipe, Process, Queue

import ijson

import smarts.core.scenario as scenario


@dataclass
class RequestHistoryRange:
    start_index: int
    batch_count: int


class Traffic_history_service:
    """responsible for dynamically fetching traffic history json to reduce
    memory use of traffic history data
    """

    class QueueDone:
        pass

    def __init__(self, history_file_path):
        self._history_file_path = history_file_path
        self._all_timesteps = set()
        self._current_traffic_history = {}
        self._prev_batch_history = {}
        # return if traffic history is not used
        if history_file_path is None:
            return

        self._log = logging.getLogger(self.__class__.__name__)
        send_data_conn, receive_data_conn = Pipe()
        self._receive_data_conn = receive_data_conn
        self._request_queue = Queue()
        self._fetch_history_proc = Process(
            target=self._fetch_history,
            args=(
                send_data_conn,
                self._request_queue,
                self._history_file_path,
            ),
        )
        self._fetch_history_proc.daemon = True
        self._fetch_history_proc.start()

        self._range_start = 0
        self._batch_size = 300
        # initialize
        with open(self._history_file_path, "rb") as f:
            for index, (t, vehicles_state) in enumerate(
                ijson.kvitems(f, "", use_float=True)
            ):
                self._all_timesteps.add(t)
                if (
                    self._range_start <= index
                    and index < self._range_start + self._batch_size
                ):
                    self._current_traffic_history[t] = vehicles_state
        self._range_start += self._batch_size
        # prepares the next batch
        self._prepare_next_batch()
        self._receive_data_conn.recv()

    def teardown(self):
        if self.is_in_use:
            self._request_queue.put(Traffic_history_service.QueueDone())
            self._request_queue.close()
            self._request_queue = None
            self._fetch_history_proc.join(timeout=3)
            if self._fetch_history_proc.is_alive():
                self._log.warning("fetch history process still alive after teardown")
            self._fetch_history_proc = None
            self._history_file_path = None

    def __del__(self):
        self.teardown()

    @property
    def is_in_use(self):
        return self._history_file_path is not None

    def _fetch_history(self, send_data_conn, request_queue, history_file_path):
        """prepare 1 batch ahead, when received request, immediately return the previously
        prepared batch and prepares the next batch.
        """
        return_batch = {}
        while True:
            historyRange = request_queue.get()
            if type(historyRange) is Traffic_history_service.QueueDone:
                break

            assert isinstance(historyRange, RequestHistoryRange)
            send_data_conn.send(return_batch)
            return_batch = {}
            with open(history_file_path, "rb") as f:
                for index, (t, vehicles_state) in enumerate(
                    ijson.kvitems(f, "", use_float=True)
                ):
                    if (
                        historyRange.start_index <= index
                        and index < historyRange.start_index + historyRange.batch_count
                    ):
                        return_batch[t] = vehicles_state
        send_data_conn.close()

    @property
    def all_timesteps(self):
        return self._all_timesteps

    @property
    def history_file_path(self):
        return self._history_file_path

    @property
    def traffic_history(self):
        return {**self._current_traffic_history, **self._prev_batch_history}

    def _prepare_next_batch(self):
        self._request_queue.put(
            RequestHistoryRange(
                start_index=self._range_start,
                batch_count=self._batch_size,
            )
        )
        self._range_start += self._batch_size

    def fetch_history_at_timestep(self, timestep: str):
        if timestep not in self._all_timesteps:
            return {}
        elif timestep in self.traffic_history:
            return self.traffic_history[timestep]

        # ask child process to prepare the next batch:
        self._prepare_next_batch()
        self._prev_batch_history = self._current_traffic_history
        # receives the previous batch child process prepared
        self._current_traffic_history = self._receive_data_conn.recv()
        if timestep in self._current_traffic_history:
            return self._current_traffic_history[timestep]
        # no history exists at requested timestamp
        return {}

    @staticmethod
    def apply_map_location_offset(position, map_offset):
        return [pos + map_offset[i] for i, pos in enumerate(position[:2])]

    @staticmethod
    def fetch_agent_missions(
        history_file_path: str, scenario_root_path: str, mapLocationOffset
    ):
        assert os.path.isdir(scenario_root_path)
        history_mission_filepath = os.path.join(
            scenario_root_path, "history_mission.pkl"
        )

        if not os.path.exists(history_mission_filepath):
            history_mission = {}
        else:
            with open(history_mission_filepath, "rb") as f:
                history_mission = pickle.load(f)

        if history_file_path in history_mission:
            return history_mission[history_file_path]

        vehicle_missions = {}
        with open(history_file_path, "rb") as f:
            for t, vehicles_state in ijson.kvitems(f, "", use_float=True):
                for vehicle_id in vehicles_state:
                    if vehicle_id in vehicle_missions:
                        continue
                    vehicle_missions[vehicle_id] = scenario.Mission(
                        start=scenario.Start(
                            Traffic_history_service.apply_map_location_offset(
                                vehicles_state[vehicle_id]["position"],
                                mapLocationOffset,
                            ),
                            scenario.Heading(vehicles_state[vehicle_id]["heading"]),
                        ),
                        goal=scenario.EndlessGoal(),
                        start_time=float(t),
                    )
        history_mission[history_file_path] = vehicle_missions

        # update cached history_mission_file
        with open(history_mission_filepath, "wb") as f:
            pickle.dump(history_mission, f)

        return vehicle_missions
