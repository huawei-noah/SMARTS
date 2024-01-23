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
import json
import os
import subprocess
from typing import Dict, Set

from smarts.core import config
from smarts.core.utils.networking import find_free_port
from smarts.core.utils.sumo_utils import SUMO_PATH


class SumoServer:
    """A centralized server for handling SUMO instances to prevent race conditions."""

    def __init__(self, host, port) -> None:
        self._host = host
        self._port = port

        self._used_ports: Set[int] = set()
        self._sumo_servers: Dict[int, subprocess.Popen[bytes]] = {}

    async def start(self):
        """Start the server."""
        # Create a socket object
        server = await asyncio.start_server(self.handle_client, self._host, self._port)
        addr = server.sockets[0].getsockname()
        print(f"Server listening on {addr}")

        async with server:
            try:
                await server.serve_forever()
            except KeyboardInterrupt:
                for port in self._sumo_servers:
                    self._handle_disconnect(port)

    def _handle_disconnect(self, port):
        self._used_ports.discard(port)
        _sumo_proc = self._sumo_servers.get(port)
        if _sumo_proc is not None:
            _sumo_proc.kill()
            _sumo_proc = None
            del self._sumo_servers[port]

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
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
                    while {port := find_free_port()} in self._used_ports:
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

    ss = SumoServer(_host, _port)
    asyncio.run(ss.start())
