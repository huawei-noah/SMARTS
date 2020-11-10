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
import time
import uuid
import json
import logging
import threading
from queue import Queue
from typing import Union
from pathlib import Path

import websocket
import numpy as np

from smarts.core.utils.file import unpack
from . import types


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
        num_retries: int = 5,
        wait_between_retries: float = 0.5,
        output_dir: str = None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        client_id = str(uuid.uuid4())[:8]

        if endpoint is None:
            endpoint = "ws://localhost:8081"

        self._logging_thread = None
        self._logging_queue = Queue()
        if output_dir:
            self._logging_thread = self._spawn_logging_thread(output_dir, client_id)
            self._logging_thread.start()

        self._state_queue = Queue()
        self._thread = self._connect(
            endpoint=f"{endpoint}/simulations/{client_id}/broadcast",
            queue=self._state_queue,
            num_retries=num_retries,
            wait_between_retries=wait_between_retries,
        )
        self._thread.start()

    def _spawn_logging_thread(self, output_dir, client_id):
        output_dir = Path(f"{output_dir}/{int(time.time())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = (output_dir / client_id).with_suffix(".jsonl")
        return threading.Thread(
            target=self._write_log_state, args=(self._logging_queue, path), daemon=True
        )

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
        num_retries: int = 5,
        wait_between_retries: float = 0.5,
    ):
        client = Client(
            endpoint=endpoint,
            num_retries=num_retries,
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
        queue,
        num_retries: int = 50,
        wait_between_retries: float = 0.05,
    ):
        threadlocal = threading.local()

        def optionally_serialize_and_write(state: Union[types.State, str], ws):
            # if not already serialized
            if not isinstance(state, str):
                state = unpack(state)
                state = json.dumps(state, cls=JSONEncoder)

            ws.send(state)

        def on_close(ws):
            self._log.debug("Connection to Envision closed")

        def on_error(ws, error):
            self._log.error(f"Connection to Envision terminated with: {error}")

        def on_open(ws):
            setattr(threadlocal, "connection_established", True)

            while True:
                state = queue.get()
                if type(state) is Client.QueueDone:
                    ws.close()
                    break

                optionally_serialize_and_write(state, ws)

        def run_socket(endpoint, num_retries, wait_between_retries, threadlocal):
            connection_established = False

            tries = 1
            while tries <= num_retries and not connection_established:
                self._log.info(
                    f"Attempting to connect to Envision tries={tries}/{num_retries}"
                )
                ws = websocket.WebSocketApp(
                    endpoint, on_error=on_error, on_close=on_close, on_open=on_open
                )
                ws.run_forever()

                connection_established = getattr(
                    threadlocal, "connection_established", False
                )
                if not connection_established:
                    tries += 1
                    time.sleep(wait_between_retries)

        return threading.Thread(
            target=run_socket,
            args=(endpoint, num_retries, wait_between_retries, threadlocal),
            daemon=True,  # If False, the proc will not terminate until this thread stops
        )

    def send(self, state: types.State):
        if self._thread.is_alive():
            self._state_queue.put(state)
        if self._logging_thread:
            self._logging_queue.put(state)

    def _send_raw(self, state: str):
        """Skip serialization if we already have serialized data. This is useful if
        we are reading from file and forwarding through the websocket.
        """
        self._state_queue.put(state)

    def teardown(self):
        self._state_queue.put(Client.QueueDone())
        self._logging_queue.put(Client.QueueDone())
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        if self._logging_thread:
            self._logging_thread.join(timeout=3)
            self._logging_thread = None
