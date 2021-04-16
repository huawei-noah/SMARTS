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

"""
Run an agent in it's own (independent) process.

What Agent code does is out of our direct control, we want to avoid any interactions with global state that might be present in the SMARTS process.

To protect and isolate Agents from any pollution of global state in the main SMARTS process, we spawn Agents in their fresh and independent python process.

This script is called from within SMARTS to instantiate a remote agent.
The protocal is as follows:

1. SMARTS calls: worker.py --port 5467 # sets a unique port per agent
2. worker.py will begin listening on port 5467.
3. SMARTS connects to (ip, 5467) as a client.
4. SMARTS calls `build()` rpc with `AgentSpec` as input.
5. worker.py recieves the `AgentSpec` instances and builds the Agent.
6. SMARTS calls `act()` rpc with observation as input and receives the actions as response from worker.py.
"""

import argparse
import importlib
import logging
import os
import signal
from concurrent import futures

import grpc

from multiprocessing import Process
from smarts.proto import agent_pb2_grpc
from smarts.rpc import agent_servicer
from typing import NamedTuple, Tuple


# Front-load some expensive imports as to not block the simulation
modules = [
    "smarts.core.utils.pybullet",
    "smarts.core.utils.sumo",
    "smarts.core.sumo_road_network",
    "numpy",
    "sklearn",
    "shapely",
    "scipy",
    "trimesh",
    "panda3d",
    "gym",
    "ray",
]

for mod in modules:
    try:
        importlib.import_module(mod)
    except ImportError:
        pass

# End front-loaded imports

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"server.py - pid({os.getpid()})")

class Connection(NamedTuple):
    """ Provides connection info for a grpc server."""

    address: Tuple[str, int] = None
    process = None
    channel = None
    stub = None

def serve(port):
    ip = "[::]"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    agent_pb2_grpc.add_AgentServicer_to_server(
        agent_servicer.AgentServicer(), server
    )
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    log.debug(f"RPC-Server - ip({ip}), port({port}), pid({os.getpid()}): Started serving.")

    def stop_server(unused_signum, unused_frame):
        server.stop(0)
        log.debug(
            f"RPC-Server - ip({ip}), port({port}), pid({os.getpid()}): Received interrupt signal."
        )

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # Wait to receive server termination signal
    server.wait_for_termination()
    log.debug(f"RPC-Server - ip({ip}), port({port}), pid({os.getpid()}): Server exited")

def spawn(target, port:int)
    """Spawn a grpc server in localhost:port ."""

    server = Process(target=target, args=(port,))
    server.start()

    addr = ("localhost", port)
    channel = get_channel(addr)
    stub = agent_pb2_grpc.AgentStub(channel)

    return Connection(
        address=addr,
        process=server,
        channel=channel,
        stub=stub
    )

def get_channel(addr):
    channel = grpc.insecure_channel(f"{addr[0]}:{addr[1]}")
    try:
        # Wait until the grpc server is ready or timeout after 30 seconds
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError:
        raise Exception("Timeout in connecting to grpc server.")
    return channel



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run an agent in an independent process.")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to listen for remote client connections.",
    )

    config = parser.parse_args()
    serve(config.port)
