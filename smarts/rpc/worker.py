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
Run an agent in it's own (independent) process, to isolate and to avoid any interactions with global states in SMARTS process.
"""

import argparse
import importlib
import grpc
import logging
import os
import signal
from concurrent import futures

from multiprocessing import Process
from smarts.proto import worker_pb2_grpc
from smarts.rpc import worker_servicer
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
log = logging.getLogger(f"worker.py - pid({os.getpid()})")


class Connection(NamedTuple):
    """ Provides connection info for a grpc server."""

    address: Tuple[str, int] = None
    process: Process = None
    channel: grpc.Channel = None
    stub: worker_pb2_grpc.WorkerStub = None


def serve(port):
    ip = "[::]"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    worker_servicer_object = worker_servicer.WorkerServicer()
    worker_pb2_grpc.add_WorkerServicer_to_server(
        worker_servicer.WorkerServicer(), server
    )
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    log.debug(
        f"rpc-worker - ip({ip}), port({port}), pid({os.getpid()}): Started serving."
    )

    def stop_server(unused_signum, unused_frame):
        worker_servicer_object.destroy()
        server.stop(0)
        log.debug(
            f"rpc-worker - ip({ip}), port({port}), pid({os.getpid()}): Received interrupt signal."
        )

    # Catch keyboard interrupt and terminate signal
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # Wait to receive server termination signal
    server.wait_for_termination()
    log.debug(f"rpc-worker - ip({ip}), port({port}), pid({os.getpid()}): Server exited")


def spawn(target, port: int):
    """Spawn a grpc server in localhost:port ."""

    server = Process(target=target, args=(port,))
    server.start()

    addr = ("localhost", port)
    channel = get_channel(addr)
    stub = worker_pb2_grpc.WorkerStub(channel)

    return Connection(
        address=addr,
        process=server,
        channel=channel,
        stub=stub,
    )


def get_channel(addr):
    """Create a channel to grpc server running at addr."""

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
