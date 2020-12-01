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
import atexit
from concurrent import futures

from threading import Lock, Thread

from .remote_agent import RemoteAgent, RemoteAgentException


class RemoteAgentBuffer:
    def __init__(self, buffer_size=3):
        """
        Args:
          buffer_size: Number of RemoteAgents to pre-initialize and keep running in the background, must be non-zero (default: 3)
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)
        self._buffer_size = buffer_size
        self._replenish_threadpool = futures.ThreadPoolExecutor()
        self._agent_buffer = [self._remote_agent_future() for _ in range(buffer_size)]

        atexit.register(self.destroy)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if atexit.unregister is not None:
            atexit.unregister(self.destroy)

        for remote_agent_future in self._agent_buffer:
            # try to cancel the future
            if not remote_agent_future.cancel():
                # We can't cancel this future, wait for it to complete
                # and terminate the agent after it's been created
                remote_agent = remote_agent_future.result()
                remote_agent.terminate()

    def _remote_agent_future(self):
        return self._replenish_threadpool.submit(RemoteAgent)

    def acquire_remote_agent(self) -> RemoteAgent:
        assert len(self._agent_buffer) == self._buffer_size

        # Check if we have any done remote agent futures.
        done_future_indices = [
            idx
            for idx, agent_future in enumerate(self._agent_buffer)
            if agent_future.done()
        ]

        if len(done_future_indices) > 0:
            # If so, prefer one of these done ones to avoid sim delays.
            future = self._agent_buffer.pop(done_future_indices[0])
        else:
            # Otherwise, we will block, waiting on a remote agent future
            self._log.debug(
                "No ready remote agents, simulation will block until one is available"
            )
            future = self._agent_buffer.pop(0)

        remote_agent = future.result(timeout=10)

        # Schedule the next remote agent and add it to the buffer.
        self._agent_buffer.append(self._remote_agent_future())
        return remote_agent
