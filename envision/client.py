# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import logging
import multiprocessing
import re
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Union

import numpy as np
import websocket

from envision import types
from smarts.core.utils.file import unpack


class JSONEncoder(json.JSONEncoder):
    """This custom encoder is to support serializing more complex data from SMARTS
    including numpy arrays, NaNs, and Infinity which don't have standarized handling
    according to the JSON spec.
    """

    def default(self, obj):
        if isinstance(obj, float):
            if np.isposinf(obj):
                obj = "Infinity"
            elif np.isneginf(obj):
                obj = "-Infinity"
            elif np.isnan(obj):
                obj = "NaN"
            return obj
        elif isinstance(obj, list):
            return [self.default(x) for x in obj]
        elif isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        elif isinstance(obj, np.ndarray):
            return self.default(obj.tolist())

        return super().default(obj)


class Client:
    """Used to push state from SMARTS to Envision server while the simulation is
    running.
    """

    class QueueDone:
        pass

    def __init__(
        self,
        endpoint: str = None,
        wait_between_retries: float = 0.5,
        output_dir: str = None,
        sim_name: str = None,
        headless: bool = False,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._headless = headless

        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-4]
        client_id = current_time
        if sim_name:
            # Set string length limit to 20 characters for client id and Envision/URL display purpose
            # Replace all special (non-alphanumeric) characters to "_" to avoid invalid key values
            sim_name = re.sub(r"\W+", "_", sim_name[:20])
            client_id = f"{sim_name}_{client_id}"

        if endpoint is None:
            endpoint = "ws://localhost:8081"

        self._logging_process = None
        if output_dir:
            output_dir = Path(f"{output_dir}/{int(time.time())}")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = (output_dir / client_id).with_suffix(".jsonl")
            self._logging_queue = multiprocessing.Queue()
            self._logging_process = multiprocessing.Process(
                target=self._write_log_state,
                args=(
                    self._logging_queue,
                    path,
                ),
            )
            self._logging_process.daemon = True
            self._logging_process.start()

        if not self._headless:
            self._state_queue = multiprocessing.Queue()
            self._process = multiprocessing.Process(
                target=self._connect,
                args=(
                    f"{endpoint}/simulations/{client_id}/broadcast",
                    self._state_queue,
                    wait_between_retries,
                ),
            )
            self._process.daemon = True
            self._process.start()

    @staticmethod
    def _write_log_state(queue, path):
        with path.open("w", encoding="utf-8") as f:
            while True:
                state = queue.get()
                if type(state) is Client.QueueDone:
                    break

                if not isinstance(state, str):
                    state = unpack(state)
                    state = json.dumps(state, cls=JSONEncoder)

                f.write(f"{state}\n")

    @staticmethod
    def read_and_send(
        path: str,
        endpoint: str = "ws://localhost:8081",
        timestep_sec: float = 0.1,
        wait_between_retries: float = 0.5,
    ):
        client = Client(
            endpoint=endpoint,
            wait_between_retries=wait_between_retries,
        )
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                time.sleep(timestep_sec)
                client._send_raw(line)

            client.teardown()
            logging.info("Finished Envision data replay")

    def _connect(
        self,
        endpoint,
        state_queue,
        wait_between_retries: float = 0.05,
    ):
        connection_established = False

        def optionally_serialize_and_write(state: Union[types.State, str], ws):
            # if not already serialized
            if not isinstance(state, str):
                state = unpack(state)
                state = json.dumps(state, cls=JSONEncoder)

            ws.send(state)

        def on_close(ws):
            self._log.debug("Connection to Envision closed")

        def on_error(ws, error):
            if str(error) == "'NoneType' object has no attribute 'sock'":
                # XXX: websocket-client library outputs some strange logs, just
                #      surpress them for now.
                return

            self._log.error(f"Connection to Envision terminated with: {error}")

        def on_open(ws):
            nonlocal connection_established
            connection_established = True

            while True:
                state = state_queue.get()
                if type(state) is Client.QueueDone:
                    ws.close()
                    break

                optionally_serialize_and_write(state, ws)

        def run_socket(endpoint, wait_between_retries):
            nonlocal connection_established
            tries = 1
            while True:
                ws = websocket.WebSocketApp(
                    endpoint, on_error=on_error, on_close=on_close, on_open=on_open
                )

                with warnings.catch_warnings():
                    # XXX: websocket-client library seems to have leaks on connection
                    #      retry that cause annoying warnings within Python 3.8+
                    warnings.filterwarnings("ignore", category=ResourceWarning)
                    ws.run_forever()

                if not connection_established:
                    self._log.info(f"Attempt {tries} to connect to Envision.")
                else:
                    # When connection lost, retry again every 3 seconds
                    wait_between_retries = 3
                    self._log.info(
                        f"Connection to Envision lost. Attempt {tries} to reconnect."
                    )

                tries += 1
                time.sleep(wait_between_retries)

        run_socket(endpoint, wait_between_retries)

    def send(self, state: types.State):
        if not self._headless and self._process.is_alive():
            self._state_queue.put(state)
        if self._logging_process:
            self._logging_queue.put(state)

    def _send_raw(self, state: str):
        """Skip serialization if we already have serialized data. This is useful if
        we are reading from file and forwarding through the websocket.
        """
        self._state_queue.put(state)

    def teardown(self):
        if not self._headless:
            self._state_queue.put(Client.QueueDone())
            self._process.join(timeout=3)
            self._process = None
            self._state_queue.close()

        if self._logging_process:
            self._logging_queue.put(Client.QueueDone())
            self._logging_process.join(timeout=3)
            self._logging_process = None
            self._logging_queue.close()
