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
"""
Run an agent in it's own (independent) process.

What Agent code does is out of our direct control, we want to avoid any interactions with global state that might be present in the SMARTS process.

To protect and isolate Agents from any pollution of global state in the main SMARTS process, we spawn Agents in their fresh and independent python process.

This script is called from within SMARTS to instantiate a remote agent.
The protocal is as follows:

1. SMARTS Calls: worker.py --port 5467 # sets a unique port per agent
2. worker.py will begin listening on port 5467.
3. SMARTS connects to (ip, port) as a client.
4. SMARTS calls `Build()` rpc with `AgentSpec` as input.
5. worker.py recieves the `AgentSpec` instances and builds the Agent.
6. SMARTS calls `Act()` rpc with observation as input and receives the actions as response from worker.py.
"""

import argparse
import cloudpickle
import grpc
import importlib
import logging
import os
import signal
import threading
import time
from concurrent import futures

from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc

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
log = logging.getLogger(f"worker.py - PID({os.getpid()})")


class AgentServicer(agent_pb2_grpc.AgentServicer):
    """Provides methods that implement functionality of Agent Servicer."""

    def __init__(self, stop_event):
        self._agent = None
        self._agent_spec = None
        self._stop_event = stop_event

    def Build(self, request, context):
        time_start = time.time()
        self._agent_spec = cloudpickle.loads(request.payload)
        pickle_load_time = time.time()
        self._agent = self._agent_spec.build_agent()
        agent_build_time = time.time()
        log.debug(
            "build agent timings:\n"
            f"  total ={agent_build_time - time_start:.2}\n"
            f"  pickle={pickle_load_time - time_start:.2}\n"
            f"  build ={agent_build_time - pickle_load_time:.2}\n"
        )
        return agent_pb2.Status(result="Success")

    def Act(self, request, context):
        if self._agent == None or self._agent_spec == None:
            return agent_pb2.Action(
                status=agent_pb2.Status(result="Remote agent not built yet.")
            )

        adapted_obs = self._agent_spec.observation_adapter(
            cloudpickle.loads(request.payload)
        )
        action = self._agent.act(adapted_obs)
        adapted_action = self._agent_spec.action_adapter(action)
        return agent_pb2.Action(
            status=agent_pb2.Status(result="Success"),
            action=cloudpickle.dumps(adapted_action),
        )

    def Stop(self, request, context):
        self._stop_event.set()
        log.debug(f"Worker - PID({os.getpid()}): Stopped by client.")
        return agent_pb2.Output()


def serve(port):
    ip = "[::]"
    stop_event = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3))
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(stop_event), server)
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    log.debug(f"Worker - {ip}, {port}, PID({os.getpid()}): Started serving.")

    def stop_server(unused_signum, unused_frame):
        stop_event.set()
        log.debug(
            f"Worker - {ip}, {port}, PID({os.getpid()}): Server stopped by interrupt signal."
        )

    # Catch keyboard interrupt
    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # Wait to receive server termination signal
    stop_event.wait()
    server.stop(0)
    log.debug(f"Worker - {ip}, {port}, PID({os.getpid()}): Server exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run an agent in an independent process.")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to listen for remote client connections.",
    )

    args = parser.parse_args()
    serve(args.port)
