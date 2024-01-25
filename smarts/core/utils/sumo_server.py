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
from asyncio.streams import StreamReader, StreamWriter
from typing import Any, Dict, Set

from smarts.core import config
from smarts.core.utils.networking import find_free_port
from smarts.core.utils.sumo_utils import SUMO_PATH


class SumoServer:
    """A centralized server for handling SUMO instances to prevent race conditions."""

    def __init__(self, host, port, reserved_count) -> None:
        self._host = host
        self._port = port

        self._used_ports: Set[int] = set()
        self._sumo_servers: Dict[int, subprocess.Popen[bytes]] = {}

        self._reserve_count: float = reserved_count
        self._server_pool: multiprocessing.Queue()

    async def start(self):
        """Start the server."""
        # Create a socket object
        server = await asyncio.start_server(self.handle_client, self._host, self._port)

        address = server.sockets[0].getsockname()
        print(f"Server listening on `{address = }`")

        resource_pooling_task = asyncio.create_task(self._resource_pooling_loop())

        async with server:
            try:
                await asyncio.gather(server.serve_forever(), resource_pooling_task)
            except KeyboardInterrupt:
                for port in self._sumo_servers:
                    self._handle_disconnect(port)

    async def _resource_pooling_loop(
        self,
    ):
        # Get the current event loop.
        loop = asyncio.get_running_loop()
        futures: Dict[Any, asyncio.Future[Any]] = {}
        while True:
            for _ in range(self._reserve_count - len(self._sumo_servers)):
                # Create a new Future object.
                fut = loop.create_future()
                while (port := find_free_port()) in self._used_ports:
                    pass
                futures[port] = fut
                loop.create_task(self.open_sumo(fut, port))

            for p, f in futures.copy().items():
                if f.done():
                    if f.cancelled():
                        del futures[p]
                        continue
                    poll, port, _sumo_process = f.result()
                    if poll is not None:
                        continue
                    print(f"sumo pooled {port = }, {poll = }")
                    self._server_pool.put((port, _sumo_process))

    async def open_sumo(self, future: asyncio.futures.Future, port: int):
        _sumo_proc = subprocess.Popen(
            [
                os.path.join(
                    SUMO_PATH,
                    "bin",
                ),
                f"--remote-port={port}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        await asyncio.sleep(0.1)
        if (poll := _sumo_proc.poll()) is None:
            self._server_pool[port] = _sumo_proc
        future.set_result((poll, port, _sumo_proc))

    def _handle_disconnect(self, port: int):
        _sumo_proc = self._sumo_servers.get(port)
        if _sumo_proc is not None:
            _sumo_proc.kill()
            _sumo_proc = None
            del self._sumo_servers[port]
        self._used_ports.discard(port)

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        """Read data from the client."""
        addr = writer.get_extra_info("peername")
        print(f"Received connection from {addr}")
        try:
            port = None
            _sumo_proc = None
            while True:
                data = await reader.readline()
                message = data.decode("utf-8")
                # print(f"Received {message!r} from {addr}")
                if message.startswith("sumo"):
                    while (port := find_free_port()) in self._used_ports:
                        pass
                    self._used_ports.add(port)
                    # Send a response back to the client
                    sumo_binary, _, sumo_cmd = message.partition(":")
                    response_list = json.loads(sumo_cmd)
                    _sumo_proc = subprocess.Popen(
                        [
                            os.path.join(SUMO_PATH, "bin", sumo_binary),
                            f"--remote-port={port}",
                            *response_list,
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                    )
                    self._sumo_servers[port] = _sumo_proc

                    response = f"{self._host}:{port}"
                    print(f"Send: {response!r}")
                    writer.write(response.encode("utf-8"))
                if message.startswith("e:"):
                    self._handle_disconnect(port)
                    break
                if len(message) == 0:
                    break

            # Close the connection
            await writer.drain()
            writer.close()
        except asyncio.CancelledError:
            # Handle client disconnect
            self._handle_disconnect(port)
            print(f"Client {addr} disconnected unexpectedly.")


if __name__ == "__main__":

    # Define the host and port on which the server will listen
    _host = config()(
        "sumo", "server_host"
    )  # Use '0.0.0.0' to listen on all available interfaces
    _port = config()("sumo", "server_port")
    _server_pool_size = config()("sumo", "server_pool_size")

    ss = SumoServer(_host, _port, _server_pool_size)
    asyncio.run(ss.start())
