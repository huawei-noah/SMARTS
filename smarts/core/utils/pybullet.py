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
import functools
import inspect

from smarts.core.utils.logging import suppress_output

# XXX: Importing pybullet logs an annoying build version tag. There's no "friendly"
#      way to fix this since they simply use print(...). Disabling logging at the
#      time of import is our hack.
with suppress_output(stderr=False):
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
        super().__del__()

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        if name in {"__deepcopy__", "__getstate__", "__setstate__"}:
            raise RuntimeError(f"{self.__class__} does not allow `{name}`")
        return super().__getattr__(name)
