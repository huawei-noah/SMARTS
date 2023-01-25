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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
import logging
import multiprocessing
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import websocket

from envision import types
from envision.client_config import EnvisionStateFilter
from envision.data_formatter import EnvisionDataFormatter, EnvisionDataFormatterArgs
from smarts.core.utils.file import unpack
from smarts.core.utils.logging import suppress_websocket


@dataclass
class JSONEncodingState:
    """This class is necessary to ensure that the custom json encoder tries to deserialize the data.
    This is vital to ensure that non-standard json literals like `Infinity`, `-Infinity`, `NaN` are not added to the output json.
    """

    data: Any


class CustomJSONEncoder(json.JSONEncoder):
    """This custom encoder is to support serializing more complex data from SMARTS
    including numpy arrays, NaNs, and Infinity which don't have standarized handling
    according to the JSON spec.
    """

    def default(self, obj):
        if isinstance(obj, JSONEncodingState):
            return self.default(unpack(obj.data))
        elif isinstance(obj, (int, float)):
            if np.isposinf(obj):
                obj = "Infinity"
            elif np.isneginf(obj):
                obj = "-Infinity"
            elif np.isnan(obj):
                obj = "NaN"
            return obj
        elif obj is None or isinstance(obj, (str, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self.default(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [self.default(x) for x in obj]
        elif isinstance(obj, Path):
            return str(obj)

        return super().default(obj)


class Client:
    """Used to push state from SMARTS to Envision server while the simulation is
    running.
    """

    class QueueDone:
        """A marker type to indicate termination of messages."""

        pass

    def __init__(
        self,
        endpoint: Optional[str] = None,
        wait_between_retries: float = 0.5,
        output_dir: Optional[str] = None,
        sim_name: Optional[str] = None,
        headless: bool = False,
        envision_state_filter: Optional[EnvisionStateFilter] = None,
        data_formatter_args: Optional[EnvisionDataFormatterArgs] = None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._headless = headless

        self._envision_state_filter = (
            envision_state_filter or EnvisionStateFilter.default()
        )
        self._data_formatter_args = data_formatter_args or EnvisionDataFormatterArgs(
            "default"
        )

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
                args=(self._logging_queue, path, self._data_formatter_args),
            )
            self._logging_process.daemon = True
            self._logging_process.start()

        self._process = None
        self._state_queue = None
        if not self._headless:
            self._state_queue = multiprocessing.Queue()
            self._process = multiprocessing.Process(
                target=self._connect,
                args=(
                    f"{endpoint}/simulations/{client_id}/broadcast",
                    self._state_queue,
                    wait_between_retries,
                    self._data_formatter_args,
                ),
            )
            self._process.daemon = True
            self._process.start()

    @property
    def headless(self):
        """Indicates if this client is disconnected from the remote."""
        return self._headless

    @property
    def envision_state_filter(self) -> EnvisionStateFilter:
        """Filtering options for data."""
        return self._envision_state_filter

    @staticmethod
    def _write_log_state(
        queue, path, data_formatter_args: Optional[EnvisionDataFormatterArgs]
    ):
        data_formatter = None
        if data_formatter_args:
            data_formatter = EnvisionDataFormatter(data_formatter_args)
        with path.open("w", encoding="utf-8") as f:
            while True:
                state = queue.get()
                if type(state) is Client.QueueDone:
                    break

                if not isinstance(state, str):
                    if data_formatter:
                        data_formatter.reset()
                        data_formatter.add(state)
                        state = data_formatter.resolve()
                    state = json.dumps(JSONEncodingState(state), cls=CustomJSONEncoder)

                f.write(f"{state}\n")

    @staticmethod
    def read_and_send(
        path: str,
        endpoint: str = "ws://localhost:8081",
        fixed_timestep_sec: float = 0.1,
        wait_between_retries: float = 0.5,
    ):
        """Send a pre-recorded envision simulation to the envision server."""

        client = Client(
            endpoint=endpoint,
            wait_between_retries=wait_between_retries,
        )
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                time.sleep(fixed_timestep_sec)
                client._send_raw(line)

            client.teardown()
            logging.info("Finished Envision data replay")

    def _connect(
        self,
        endpoint,
        state_queue,
        wait_between_retries: float = 0.05,
        data_formatter_args: Optional[EnvisionDataFormatterArgs] = None,
    ):
        connection_established = False
        warned_about_connection = False
        data_formatter = None
        if data_formatter_args:
            data_formatter = EnvisionDataFormatter(data_formatter_args)

        def optionally_serialize_and_write(state: Union[types.State, str], ws):
            # if not already serialized
            if not isinstance(state, str):
                if data_formatter:
                    data_formatter.reset()
                    data_formatter.add(state)
                    state = data_formatter.resolve()
                state = json.dumps(JSONEncodingState(state), cls=CustomJSONEncoder)

            ws.send(state)

        def on_close(ws, code=None, reason=None):
            self._log.debug("Connection to Envision closed")

        def on_error(ws, error):
            nonlocal connection_established, warned_about_connection
            if str(error) == "'NoneType' object has no attribute 'sock'":
                # XXX: websocket-client library outputs some strange logs, just
                #      surpress them for now.
                return

            if connection_established:
                self._log.error(f"Connection to Envision terminated with: {error}")
            else:
                logmsg = f"Connection to Envision failed with: {error}."
                if not warned_about_connection:
                    self._log.warning(
                        logmsg
                        + "  Try using `--headless` if not using Envision server."
                    )
                    warned_about_connection = True
                else:
                    self._log.info(logmsg)

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
                # TODO: use a real network socket instead (probably UDP)
                ws = websocket.WebSocketApp(
                    endpoint, on_error=on_error, on_close=on_close, on_open=on_open
                )

                with suppress_websocket():
                    ws.run_forever()

                if not connection_established:
                    self._log.info(f"Attempt {tries} to connect to Envision.")
                else:
                    # No information left to send, connection is likely done
                    if state_queue.empty():
                        break
                    # When connection lost, retry again every 3 seconds
                    wait_between_retries = 3
                    self._log.info(
                        f"Connection to Envision lost. Attempt {tries} to reconnect."
                    )

                tries += 1
                time.sleep(wait_between_retries)

        run_socket(endpoint, wait_between_retries)

    def send(self, state: Union[types.State, types.Preamble]):
        """Send the given envision state to the remote as the most recent state."""
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
        """Clean up the client resources."""
        if self._state_queue:
            self._state_queue.put(Client.QueueDone())

        if self._process:
            self._process.join(timeout=3)
            self._process = None

        if self._state_queue:
            self._state_queue.close()
            self._state_queue = None

        if self._logging_process and self._logging_queue:
            self._logging_queue.put(Client.QueueDone())
            self._logging_process.join(timeout=3)
            self._logging_process = None
            self._logging_queue.close()
            self._logging_queue = None
