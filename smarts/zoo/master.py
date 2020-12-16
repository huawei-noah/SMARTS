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
import argparse
import grpc
import logging
import pathlib
import signal
import subprocess
import sys
from concurrent import futures
from multiprocessing import Process

from smarts.core.utils.networking import find_free_port
from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"Master")


def spawn_networked_agent(port):
    cmd = [
        sys.executable,  # path to the current python binary
        str((pathlib.Path(__file__).parent / "worker.py").absolute().resolve()),
        "--port",
        str(port),
    ]

    agent_proc = subprocess.Popen(cmd)
    return agent_proc


class AgentServicer(agent_pb2_grpc.AgentServicer):
    """Provides methods that implement functionality of Zoo Master."""

    def __init__(self):
        self.agent_procs = []

    def __del__(self):
        # cleanup
        log.debug("Cleaning up zoo workers")
        for proc in self.agent_procs:
            proc.kill()
            proc.wait()

    def SpawnWorker(self, request, context):
        port = find_free_port()

        proc = spawn_networked_agent(port)
        if proc:
            self.agent_procs.append(proc)
            return agent_pb2.Connection(
                status=agent_pb2.Status(result="success"), port=port
            )

        return agent_pb2.Connection(status=agent_pb2.Status(result="error"), port=port)

    def TestConnection(self, request, context):
        print(f"Input: {request.msg}, Note: Inside TestConnection() RPC.")
        return agent_pb2.Output(msg=f"Input: {request.msg}, Output: Success.")


def serve(port):
    ip = "[::]"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    agent_pb2_grpc.add_AgentServicer_to_server(AgentServicer(), server)
    server.add_insecure_port(f"{ip}:{port}")
    server.start()

    log.debug(f"Master: Started serving at {ip}, {port}")
    print(f"Master: Started serving at {ip}, {port}")

    def stop_server(unused_signum, unused_frame):
        server.stop(0)
        log.debug("GRPC server stopped by interrupt signal.")

    # Catch keyboard interrupt
    signal.signal(signal.SIGINT, stop_server)

    server.wait_for_termination()
    log.debug("Server exited.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Listens for requests to allocate agents and executes them on-demand"
    )
    parser.add_argument(
        "--port", type=int, default=7432, help="Port to listen on",
    )

    args = parser.parse_args()
    serve(args.port)
