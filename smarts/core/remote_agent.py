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
import time

from concurrent import futures

from smarts.core.agent import AgentSpec
from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, master_address, worker_address):
        self._log = logging.getLogger(self.__class__.__name__)

        self.last_act_future = None

        self.master_ip, self.master_port = master_address
        self.master_channel = grpc.insecure_channel(
            f"{self.master_ip}:{self.master_port}"
        )
        self.worker_ip, self.worker_port = worker_address
        self.worker_channel = grpc.insecure_channel(
            f"{self.worker_ip}:{self.worker_port}"
        )
        try:
            # Wait until the grpc server is ready or timeout after 30 seconds
            grpc.channel_ready_future(self.master_channel).result(timeout=30)
            grpc.channel_ready_future(self.worker_channel).result(timeout=30)
        except grpc.FutureTimeoutError as e:
            raise RemoteAgentException(
                "Timeout while connecting to remote worker process."
            ) from e
        self.master_stub = agent_pb2_grpc.AgentStub(self.master_channel)
        self.worker_stub = agent_pb2_grpc.AgentStub(self.worker_channel)

    def _act(self, obs):
        try:
            response_future = self.worker_stub.Act.future(
                agent_pb2.Observation(payload=cloudpickle.dumps(obs))
            )
        except grpc.RpcError as e:
            self.terminate()
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise RemoteAgentException("Remote worker process is not avaliable.") from e
            else:
                raise RemoteAgentException(
                    "Error in retrieving agent action from remote worker process."
                ) from e

        return response_future

    def act(self, obs):
        # Run task asynchronously and return a Future.
        # Keep track of last action future returned.
        self.last_act_future = self._act(obs)
        return self.last_act_future

    def start(self, agent_spec: AgentSpec):
        # Send the AgentSpec to the agent runner
        # Cloudpickle used only for the agent_spec to allow for serialization of lambdas
        self.worker_stub.Build(
            agent_pb2.Specification(payload=cloudpickle.dumps(agent_spec))
        )

    def terminate(self):
        # If the last action future returned is incomplete, cancel it first.
        if (self.last_act_future != None) and (not self.last_act_future.done()):
            self.last_act_future.cancel()
            self._log.debug(
                f"remote_agent.py::terminate(), last action future status = {self.last_act_future.running()} = ({self.worker_ip},{self.worker_port})"
            )

        # Close worker channel
        self.worker_channel.close()
        # Stop the remote worker process
        response = self.master_stub.StopWorker(agent_pb2.Port(num=self.worker_port))
        if response.code != 0:
            raise RemoteAgentException(
                f"Trying to stop worker process with invalid address ({self.worker_ip}, {self.worker_port})."
            )
        # Close master channel
        self.master_channel.close()
