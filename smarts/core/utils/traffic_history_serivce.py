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
from multiprocessing import Process, Queue, Pipe
import ijson

import smarts.core.scenario as scenario

class Traffic_history_service:
    """ responsible for dynamically fetching traffic history json to reduce
    memory use of traffic history data 
    """
    
    def __init__(self, history_file_path):
        self._history_file_path = history_file_path
        self._all_timesteps = set()
        self._current_traffic_history = {}
        # return if traffic history is not used
        if history_file_path is None:
            return

        self._range_start = 0
        self._range_end = 300
        # initialize
        with open(self._history_file_path, 'rb') as f:
            for index, time_and_data_tuple in enumerate(ijson.kvitems(f, '', use_float=True)):
                self._all_timesteps.add(time_and_data_tuple[0])
                if self._range_start <= index and index <= self._range_end:
                    self._current_traffic_history[time_and_data_tuple[0]] = time_and_data_tuple[1]

        send_data_conn, receive_data_conn = Pipe()
        self._receive_data_conn = receive_data_conn
        self._request_queue = Queue()
        self._fetch_history_proc = Process(
            target=self._fetch_history,
            args=(
                send_data_conn,
                self._request_queue,
                self._history_file_path,
                len(self._all_timesteps)
            ),
        )
        self._fetch_history_proc.daemon = True
        self._fetch_history_proc.start()


    def _fetch_history(self, send_data, request_queue, history_file_path, total_timestep_size):
        return_batch = {}
        while True:
            request_timestamp = request_queue.get()
            send_data.send(return_batch)
            return_batch = {}
            # print(f"received request: {request_timestamp}")
            with open(history_file_path, 'rb') as f:
                for t, vehicles_state in ijson.kvitems(f, '', use_float=True):
                    if t:
                        return_batch[t] = vehicles_state
        send_data.close()

    @property
    def all_timesteps(self):
        return self._all_timesteps

    @property
    def current_traffic_history(self):
        return self._current_traffic_history

    def fetch_history_at_timestep(self, timestep):
        print(timestep)
        if timestep not in self._all_timesteps:
            return {}
        elif timestep in self._current_traffic_history:
            return self._current_traffic_history[timestep]

        # load from child process
        # initially Have child process load the first batch and, and prepare the next batch, 
        # Then when next request, send back the prepared batch, and get the next batch


        # ask fetch_history worker to prepare the data:
        self._request_queue.put(timestep)
        timestep_data = self._receive_data_conn.recv()
        return timestep_data

    def fetch_agent_missions(self):
        vehicle_missions = {}
        with open(self._history_file_path, 'rb') as f:
            for t, vehicles_state in ijson.kvitems(f, '', use_float=True):
                for vehicle_id in vehicles_state:
                    if vehicle_id in vehicle_missions:
                        continue
                    vehicle_missions[vehicle_id] = scenario.Mission(
                        start=scenario.Start(
                            vehicles_state[vehicle_id]["position"][:2],
                            scenario.Heading(vehicles_state[vehicle_id]["heading"]),
                        ),
                        goal=scenario.EndlessGoal(),
                        start_time=float(t),
                    )

        return vehicle_missions
        
        


        
