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
import grpc
import logging
import os
import time

from smarts.core.utils.networking import find_free_port
from smarts.zoo import worker_pb2
from smarts.zoo import worker_pb2_grpc
from smarts.zoo import worker as zoo_worker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"worker_servicer.py - pid({os.getpid()})")


class WorkerServicer(worker_pb2_grpc.WorkerServicer):
    """Provides methods that implement functionality of Worker Servicer."""

    def __init__(self):
        self._agent = None
        self._agent_spec = None

    def build(self, request, context):
        time_start = time.time()
        self._agent_spec = cloudpickle.loads(request.payload)
        pickle_load_time = time.time()
        self._agent = self._agent_spec.build_agent()
        agent_build_time = time.time()
        log.debug(
            "Build agent timings:\n"
            f"  total ={agent_build_time - time_start:.2}\n"
            f"  pickle={pickle_load_time - time_start:.2}\n"
            f"  build ={agent_build_time - pickle_load_time:.2}\n"
        )
        return worker_pb2.Status()

    def act(self, request, context):
        if self._agent == None or self._agent_spec == None:
            context.set_details(f"Remote agent not built yet.")
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return worker_pb2.Action()

        adapted_obs = self._agent_spec.observation_adapter(
            cloudpickle.loads(request.payload)
        )
        action = self._agent.act(adapted_obs)
        adapted_action = self._agent_spec.action_adapter(action)
        return worker_pb2.Action(action=cloudpickle.dumps(adapted_action))
