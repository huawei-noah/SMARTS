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

import logging
import os
import pathlib
import subprocess
import sys
import time

import cloudpickle
import grpc

from smarts.core.utils.networking import find_free_port
from smarts.zoo import manager_pb2, manager_pb2_grpc
from smarts.zoo import worker as zoo_worker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"manager_servicer.py - pid({os.getpid()})")


class ManagerServicer(manager_pb2_grpc.ManagerServicer):
    """Provides methods that implement functionality of Manager Servicer."""

    def __init__(self):
        self._workers = {}

    def __del__(self):
        self.destroy()

    def spawn_worker(self, request, context):
        port = find_free_port()

        cmd = [
            sys.executable,  # Path to the current Python binary.
            str((pathlib.Path(__file__).parent / "worker.py").absolute().resolve()),
            "--port",
            str(port),
        ]

        worker = subprocess.Popen(cmd)
        if worker.poll() == None:
            self._workers[port] = worker
            return manager_pb2.Port(num=port)

        context.set_details("Error in spawning worker subprocess.")
        context.set_code(grpc.StatusCode.INTERNAL)
        return manager_pb2.Port()

    def stop_worker(self, request, context):
        log.debug(
            f"Manager - pid({os.getpid()}), received stop signal for worker at port {request.num}."
        )

        # Get worker_process corresponding to the received port number.
        worker = self._workers.get(request.num, None)
        if worker == None:
            context.set_details(
                f"Trying to stop nonexistent worker with a port {request.num}."
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return manager_pb2.Status()

        # Terminate worker process.
        worker.terminate()
        worker.wait()

        # Delete worker process entry from dictionary.
        del self._workers[request.num]

        return manager_pb2.Status()

    def destroy(self):
        log.debug(
            f"Manager - pid({os.getpid()}), shutting down remaining agent worker processes."
        )
        for worker in self._workers.values():
            if worker.poll() == None:
                worker.terminate()
                worker.wait()
