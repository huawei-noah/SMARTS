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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from smarts.core.utils.core_logging import suppress_output

# XXX: Importing pybullet logs an annoying build version tag. There's no "friendly"
#      way to fix this since they simply use print(...). Disabling logging at the
#      time of import is our hack.
with suppress_output():
    import pybullet
    from pybullet import *
    from pybullet_utils import bullet_client


class SafeBulletClient(bullet_client.BulletClient):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None):
        """Creates a Bullet client and connects to a simulation.

        Args:
          connection_mode:
            `None` connects to an existing simulation or, if fails, creates a
              new headless simulation,
            `pybullet.GUI` creates a new simulation with a GUI,
            `pybullet.DIRECT` creates a headless simulation,
            `pybullet.SHARED_MEMORY` connects to an existing simulation.
        """
        with suppress_output(stderr=False):
            super().__init__(connection_mode=connection_mode)

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            super().__del__()
        except TypeError as error:
            # Pybullet 3.2.6 currently attempts to catch an error type that does not exist.
            if not error.args[0].contains("BaseException"):
                raise

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        if name in {"__deepcopy__", "__getstate__", "__setstate__"}:
            raise RuntimeError(f"{self.__class__} does not allow `{name}`")
        return super().__getattr__(name)


class BulletClientMACOS:
    """This wrapper class is a hack for `macOS` where running PyBullet in GUI mode,
    alongside Panda3D segfaults. It seems due to running two OpenGL applications
    in the same process. Here we spawn a process to run PyBullet and forward all
    calls to it over unix pipes.

    N.B. This class can be directly subbed-in for pybullet_utils's BulletClient
    but your application must start from a,

        if __name__ == "__main__:":
            # https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html
            import multiprocessing as mp
            mp.set_start_method('spawn', force=True)
    """

    def __init__(self, bullet_connect_mode=pybullet.GUI):
        self._parent_conn, self._child_conn = Pipe()
        self.process = Process(
            target=BulletClientMACOS.consume,
            args=(
                bullet_connect_mode,
                self._child_conn,
            ),
        )
        self.process.start()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            self._parent_conn.send((name, args, kwargs))
            return self._parent_conn.recv()

        return wrapper

    @staticmethod
    def consume(bullet_connect_mode, connection: Connection):
        """Builds a child pybullet process.
        Args:
            bullet_connect_mode: The type of bullet process.
            connection: The child end of a pipe.
        """
        # runs in sep. process
        client = SafeBulletClient(bullet_connect_mode)

        while True:
            method, args, kwargs = connection.recv()
            result = getattr(client, method)(*args, **kwargs)
            connection.send(result)
