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

import asyncio
import asyncio.streams
import json
import multiprocessing
import os
import subprocess
import time
from asyncio.streams import StreamReader, StreamWriter
from typing import Any, Dict, List, Set

from smarts.core import config
from smarts.core.utils.file import make_dir_in_smarts_log_dir
from smarts.core.utils.networking import find_free_port
from smarts.core.utils.sumo_utils import SUMO_PATH


class SumoServer:
    """A centralized server for handling SUMO instances to prevent race conditions."""

    def __init__(self, host, port, reserved_count) -> None:
        self._host = host
        self._port = port

        self._used_ports: Set[int] = set()
        self._reserve_count: float = reserved_count

    async def start(self):
        """Start the server."""
        # Create a socket object
        server = await asyncio.start_server(self.handle_client, self._host, self._port)

        address = server.sockets[0].getsockname()
        print(f"Server listening on `{address = }`")

        async with server:
            await server.serve_forever()

    async def _process_manager(self, binary, args, f: asyncio.Future):
        """Manages the lifecycle of the TraCI server."""
        _sumo_proc = None
        try:
            # Create a new Future object.
            while (port := find_free_port()) in self._used_ports:
                pass
            self._used_ports.add(port)
            _sumo_proc = await asyncio.create_subprocess_exec(
                *[
                    os.path.join(SUMO_PATH, "bin", binary),
                    f"--remote-port={port}",
                ],
                *args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            f.set_result(
                {
                    "port": port,
                    "future": future,
                }
            )

            result = await asyncio.wait_for(future, None)
        finally:
            if port is not None:
                self._used_ports.discard(port)
            if _sumo_proc is not None and _sumo_proc.returncode is None:
                _sumo_proc.kill()

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        """Read data from the client."""
        address = writer.get_extra_info("peername")
        print(f"Received connection from {address}")
        loop = asyncio.get_event_loop()
        try:
            port = None
            _traci_server_info = None
            while True:
                data = await reader.readline()
                message = data.decode("utf-8")
                # print(f"Received {message!r} from {addr}")
                if message.startswith("sumo"):
                    kill_f = loop.create_future()
                    sumo_binary, _, sumo_cmd = message.partition(":")
                    response_list = json.loads(sumo_cmd)
                    command_args = [
                        *response_list,
                    ]
                    asyncio.create_task(
                        self._process_manager(sumo_binary, command_args, kill_f)
                    )
                    if _traci_server_info is not None:
                        print("Duplicate start request received.")
                        continue
                    _traci_server_info = await asyncio.wait_for(kill_f, None)
                    port = _traci_server_info["port"]

                    response = f"{self._host}:{port}"
                    print(f"Send TraCI address: {response!r}")
                    writer.write(response.encode("utf-8"))

                if message.startswith("e:"):
                    if _traci_server_info is None:
                        print("Kill received for uninitialized process.")
                    else:
                        _traci_server_info["future"].set_result("kill")
                    break
                if len(message) == 0:
                    break

            # Close the connection
            await writer.drain()
            writer.close()
        except asyncio.CancelledError:
            # Handle client disconnect
            pass


if __name__ == "__main__":

    # Define the host and port on which the server will listen
    _host = config()(
        "sumo", "server_host"
    )  # Use '0.0.0.0' to listen on all available interfaces
    _port = config()("sumo", "server_port")
    _server_pool_size = 2
    config()("sumo", "server_pool_size")

    ss = SumoServer(_host, _port, _server_pool_size)
    asyncio.run(ss.start())
