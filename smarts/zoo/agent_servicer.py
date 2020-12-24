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

import cloudpickle
import logging
import os
import time

from multiprocessing import Process

from smarts.core.utils.networking import find_free_port
from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc
from smarts.zoo import worker as zoo_worker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"agent_servicer.py - PID({os.getpid()})")


class AgentServicer(agent_pb2_grpc.AgentServicer):
    """Provides methods that implement functionality of Agent Servicer."""

    def __init__(self):
        self._agent = None
        self._agent_spec = None
        self._workers = {}

    def __del__(self):
        self.destroy()

    def SpawnWorker(self, request, context):
        port = find_free_port()
        proc = Process(target=zoo_worker.serve, args=(port,))
        proc.start()
        if proc.is_alive():
            self._workers[port] = proc
            return agent_pb2.Connection(
                status=agent_pb2.Status(code=0, msg="Success"), port=port
            )

        return agent_pb2.Connection(status=agent_pb2.Status(code=1, msg="Error"))

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
        return agent_pb2.Status(code=0, msg="Success")

    def Act(self, request, context):

        # def on_rpc_done():
        #     # print(f"@@@@@ agent_servicer.py, Act, on_rpc_done - PID({os.getpid()})")
        #     pass

        # context.add_callback(on_rpc_done)

        if self._agent == None or self._agent_spec == None:
            return agent_pb2.Action(
                status=agent_pb2.Status(code=1, msg="Remote agent not built yet.")
            )

        adapted_obs = self._agent_spec.observation_adapter(
            cloudpickle.loads(request.payload)
        )
        action = self._agent.act(adapted_obs)
        adapted_action = self._agent_spec.action_adapter(action)
        return agent_pb2.Action(
            status=agent_pb2.Status(code=0, msg="Success"),
            action=cloudpickle.dumps(adapted_action),
        )

    def StopWorker(self, request, context):
        print(f"Master at PID({os.getpid()}) received stop signal for worker at port {request.num}.")

        # Get worker_process corresponding to the received port number
        worker_proc = self._workers.get(request.num, None)
        if worker_proc == None:
            return agent_pb2.Status(code=1, msg=f"Error: No such worker with a port {request.num} exists.")
        # Terminate worker process
        worker_proc.terminate()
        worker_proc.join()
        # Delete worker process entry from dictionary
        del self._workers[request.num]

        return agent_pb2.Status(code=0, msg="Success")

    def destroy(self):
        print("Shutting down agent worker processes.")
        for proc in self._workers.values():
            if proc.is_alive():
                proc.terminate()
                proc.join()
