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
from typing import Optional

from smarts.core.agent_buffer import AgentBuffer
from smarts.core.buffer_agent import BufferAgent
from smarts.core.remote_agent import RemoteAgentException


class HeterogenousAgentBuffer(AgentBuffer):
    """A buffer that manages social agents."""

    def __init__(self, **kwargs):
        try:
            from smarts.core.remote_agent_buffer import RemoteAgentBuffer

            self._agent_buffer = RemoteAgentBuffer(
                timeout=0.05, zoo_manager_addrs=kwargs.get("zoo_manager_addrs")
            )

        except (ImportError, ModuleNotFoundError, RemoteAgentException):
            from smarts.core.local_agent_buffer import LocalAgentBuffer

            self._agent_buffer = LocalAgentBuffer()
        self._backup_buffer = None

    @property
    def backup_buffer(self):
        """The buffer fallback if the regular buffer fails."""
        if self._backup_buffer == None:
            from smarts.core.local_agent_buffer import LocalAgentBuffer

            self._backup_buffer = LocalAgentBuffer()
        return self._backup_buffer

    def destroy(self):
        self._agent_buffer.destroy()
        if self._backup_buffer is not None:
            self._backup_buffer.destroy()

    def acquire_agent(self, timeout: Optional[float] = None) -> BufferAgent:
        try:
            return self._agent_buffer.acquire_agent(timeout=timeout)
        except RemoteAgentException:
            return self.backup_buffer.acquire_agent(timeout=timeout)
