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
import time
from concurrent import futures

import cloudpickle
import grpc

from smarts.core.agent import AgentSpec
from smarts.zoo import manager_pb2, manager_pb2_grpc, worker_pb2, worker_pb2_grpc


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, manager_address, worker_address):
        self._log = logging.getLogger(self.__class__.__name__)

        # Track the last action future.
        self._act_future = None

        self._manager_channel = grpc.insecure_channel(
            f"{manager_address[0]}:{manager_address[1]}"
        )
        self._worker_address = worker_address
        self._worker_channel = grpc.insecure_channel(
            f"{worker_address[0]}:{worker_address[1]}"
        )
        try:
            # Wait until the grpc server is ready or timeout after 30 seconds.
            grpc.channel_ready_future(self._manager_channel).result(timeout=30)
            grpc.channel_ready_future(self._worker_channel).result(timeout=30)
        except grpc.FutureTimeoutError as e:
            raise RemoteAgentException(
                "Timeout while connecting to remote worker process."
            ) from e
        self._manager_stub = manager_pb2_grpc.ManagerStub(self._manager_channel)
        self._worker_stub = worker_pb2_grpc.WorkerStub(self._worker_channel)

    def act(self, obs):
        # Run task asynchronously and return a Future.
        self._act_future = self._worker_stub.act.future(
            worker_pb2.Observation(payload=cloudpickle.dumps(obs))
        )

        return self._act_future

    def start(self, agent_spec: AgentSpec):
        # Send the AgentSpec to the agent runner.
        # Cloudpickle used only for the agent_spec to allow for serialization of lambdas.
        self._worker_stub.build(
            worker_pb2.Specification(payload=cloudpickle.dumps(agent_spec))
        )

    def terminate(self):
        # If the last action future returned is incomplete, cancel it first.
        if (self._act_future is not None) and (not self._act_future.done()):
            self._act_future.cancel()

        # Stop the remote worker process
        response = self._manager_stub.stop_worker(
            manager_pb2.Port(num=self._worker_address[1])
        )

        # Close manager channel
        self._manager_channel.close()
