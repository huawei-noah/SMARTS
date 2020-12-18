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
from multiprocessing.connection import Client

from .agent import AgentSpec
from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, address):
        self._log = logging.getLogger(self.__class__.__name__)

        self._tp_exec = futures.ThreadPoolExecutor()

        ip, port = address
        self.channel = grpc.insecure_channel(f"{ip}:{port}")
        try:
            # Wait until the grpc server is ready or timeout after 30 seconds
            grpc.channel_ready_future(self.channel).result(timeout=30)
        except grpc.FutureTimeoutError as e:
            raise RemoteAgentException(
                "Timeout while connecting to remote worker process."
            ) from e
        self.stub = agent_pb2_grpc.AgentStub(self.channel)

    def _act(self, obs, timeout):
        try:
            response = self.stub.Act(
                agent_pb2.Observation(payload=cloudpickle.dumps(obs)), timeout=timeout
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                log.debug("Remote worker process exceeded response deadline.")
                return None
            else:
                raise RemoteAgentException(
                    f"Error in retrieving agent action from remote worker process."
                ) from e
        return cloudpickle.loads(response.action)

    def act(self, obs, timeout=None):
        # Run task asynchronously and return a Future
        return self._tp_exec.submit(self._act, obs, timeout)

    def start(self, agent_spec: AgentSpec):
        # Send the AgentSpec to the agent runner
        # Cloudpickle used only for the agent_spec to allow for serialization of lambdas
        self.stub.Build(agent_pb2.Specification(payload=cloudpickle.dumps(agent_spec)))

    def terminate(self):
        # Stop the remote worker process
        try:
            self.stub.Stop(agent_pb2.Input())
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                # Server-shutdown rpc executed. Some data transmitted but connection
                # breaks due to server shutting down. Hence, server `UNAVAILABLE`
                # error is thrown. This error can be ignored.
                pass
            else:
                raise e
        # Close the channel
        self.channel.close
        # Shutdown thread pool executor
        self._tp_exec.shutdown()
