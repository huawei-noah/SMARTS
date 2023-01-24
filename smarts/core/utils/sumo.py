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
"""Importing this module "redirects" the import to the "real" sumolib. This is available
for convenience and to reduce code duplication as sumolib lives under SUMO_HOME.
"""

import functools
import inspect
import logging
import multiprocessing
import os
import subprocess
import sys
from typing import Any, List, Optional

from smarts.core.utils import networking
from smarts.core.utils.logging import suppress_output

try:
    import sumo

    SUMO_PATH = sumo.SUMO_HOME
    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
except ImportError:
    if "SUMO_HOME" not in os.environ:
        raise ImportError("SUMO_HOME not set, can't import sumolib")
    SUMO_PATH = os.environ["SUMO_HOME"]

tools_path = os.path.join(SUMO_PATH, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)


import sumo.tools.sumolib as sumolib
import sumo.tools.traci as traci


class DomainWrapper:
    """Wraps `traci.Domain` type for the `TraciConn` utility"""

    def __init__(self, sumo_proc, domain: traci.domain.Domain) -> None:
        self._domain = domain
        self._sumo_proc = sumo_proc

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._domain, name)

        if inspect.isbuiltin(attribute) or inspect.ismethod(attribute):
            attribute = functools.partial(
                _wrap_traci_method, method=attribute, sumo_process=self._sumo_proc
            )

        return attribute


class TraciConn:
    """A simplified utility for connecting to a SUMO process."""

    def __init__(
        self,
        sumo_port: Optional[int],
        base_params: List[str],
        sumo_binary: str = "sumo",  # Literal["sumo", "sumo-gui"]
    ):
        self._sumo_proc = None
        self._traci_conn = None
        self._sumo_port = None
        self._sumo_version = ()

        if sumo_port is None:
            sumo_port = networking.find_free_port()
        self._sumo_port = sumo_port
        sumo_cmd = [
            os.path.join(SUMO_PATH, "bin", sumo_binary),
            f"--remote-port={sumo_port}",
            *base_params,
        ]

        logging.debug("Starting sumo process:\n\t %s", sumo_cmd)
        self._sumo_proc = subprocess.Popen(
            sumo_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )

    def __del__(self) -> None:
        self.close_traci_and_pipes()

    def connect(
        self,
        timeout: float = 5,
        minimum_traci_version=20,
        minimum_sumo_version=(
            1,
            10,
            0,
        ),
    ):
        """Attempt a connection with the SUMO process."""
        traci_conn = None
        try:
            with suppress_output(stdout=False):
                traci_conn = traci.connect(
                    self._sumo_port,
                    numRetries=max(0, int(20 * timeout)),
                    proc=self._sumo_proc,
                    waitBetweenRetries=0.05,
                )  # SUMO must be ready within timeout seconds
        # We will retry since this is our first sumo command
        except traci.exceptions.FatalTraCIError:
            logging.debug("TraCI could not connect in time.")
            raise
        except traci.exceptions.TraCIException:
            logging.error("SUMO process died.")
            raise
        except ConnectionRefusedError:
            logging.error(
                "Connection refused. Tried to connect to unpaired TraCI client."
            )
            raise

        try:
            vers, vers_str = traci_conn.getVersion()
            assert (
                vers >= minimum_traci_version
            ), f"TraCI API version must be >= {minimum_traci_version}. Got version ({vers})"
            self._sumo_version = tuple(
                int(v) for v in vers_str.partition(" ")[2].split(".")
            )  # e.g. "SUMO 1.11.0" -> (1, 11, 0)
            assert (
                self._sumo_version >= minimum_sumo_version
            ), f"SUMO version must be >= SUMO {minimum_sumo_version}"
        except traci.exceptions.FatalTraCIError as err:
            logging.debug("TraCI could not connect in time.")
            # XXX: the error type is changed to TraCIException to make it consistent with the
            # process died case of `traci.connect`.
            raise traci.exceptions.TraCIException(err)
        except AssertionError:
            self.close_traci_and_pipes()
            raise
        self._traci_conn = traci_conn

    @property
    def connected(self):
        """Check if the connection is still valid."""
        return self._sumo_proc is not None and self._traci_conn is not None

    @property
    def viable(self):
        """If making a connection to the sumo process is still viable."""
        return self._sumo_proc is not None and self._sumo_proc.poll() is None

    def __getattr__(self, name: str) -> Any:
        if not self.connected:
            return None

        attribute = getattr(self._traci_conn, name)

        if inspect.isbuiltin(attribute) or inspect.ismethod(attribute):
            attribute = functools.partial(
                _wrap_traci_method, method=attribute, sumo_process=self
            )

        if isinstance(attribute, traci.domain.Domain):
            attribute = DomainWrapper(sumo_proc=self, domain=attribute)

        return attribute

    def must_reset(self):
        """If the version of sumo will have errors if just reloading such that it must be reset."""
        return self._sumo_version > (1, 12, 0)

    def close_traci_and_pipes(self):
        """Safely closes all connections. We should expect this method to always work without throwing"""

        def __safe_close(conn):
            try:
                conn.close()
            except (subprocess.SubprocessError, multiprocessing.ProcessError):
                # Subprocess or process failed
                pass
            except traci.exceptions.FatalTraCIError:
                # TraCI connection is already dead.
                pass
            except AttributeError:
                # Socket was destroyed internally by a fatal error somehow.
                pass

        if self._traci_conn:
            __safe_close(self._traci_conn)

        if self._sumo_proc:
            __safe_close(self._sumo_proc.stdin)
            __safe_close(self._sumo_proc.stdout)
            __safe_close(self._sumo_proc.stderr)
            self._sumo_proc.kill()

        self._sumo_proc = None
        self._traci_conn = None

    def teardown(self):
        """Clean up all resources."""
        self.close_traci_and_pipes()


def _wrap_traci_method(*args, method, sumo_process: TraciConn, **kwargs):
    # Argument order must be `*args` first so keyword arguments are required for `method` and `sumo_process`.
    try:
        return method(*args, **kwargs)
    except traci.exceptions.FatalTraCIError:
        # Traci cannot continue
        sumo_process.close_traci_and_pipes()
        raise
    except traci.exceptions.TraCIException:
        # Case where SUMO can continue
        # TAI: consider closing the process even with a non fatal error
        raise
