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
from __future__ import annotations

import functools
import inspect
import logging
import multiprocessing
import os
import subprocess
import sys
from typing import Any, List, Literal, Optional, Tuple

from smarts.core.utils import networking
from smarts.core.utils.core_logging import suppress_output

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

try:
    import sumo.tools.sumolib as sumolib
    import sumo.tools.traci as traci
except ModuleNotFoundError as e:
    raise ImportError(
        "Missing dependencies for SUMO. Install them using the command `pip install -e .[sumo]` at the source directory."
    ) from e


class DomainWrapper:
    """Wraps `traci.Domain` type for the `TraciConn` utility"""

    def __init__(self, traci_conn, domain: traci.domain.Domain, attribute_name) -> None:
        self._domain = domain
        self._traci_conn = traci_conn
        self._attribute_name = attribute_name

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._domain, name)

        if inspect.isbuiltin(attribute) or inspect.ismethod(attribute):
            attribute = functools.partial(
                _wrap_traci_method,
                method=attribute,
                traci_conn=self._traci_conn,
                attribute_name=self._attribute_name,
            )

        return attribute


class TraciConn:
    """A simplified utility for connecting to a SUMO process."""

    def __init__(
        self,
        sumo_port: Optional[int],
        base_params: List[str],
        sumo_binary: Literal[
            "sumo", "sumo-gui"
        ] = "sumo",  # Literal["sumo", "sumo-gui"]
        host: str = "localhost",
        name: str = "",
    ):
        self._sumo_proc = None
        self._traci_conn = None
        self._sumo_port = None
        self._sumo_version: Tuple[int, ...] = tuple()
        self._host = host
        self._name = name
        self._log = logging.Logger(self.__class__.__name__)
        self._connected = False

        if sumo_port is None:
            sumo_port = networking.find_free_port()
        self._sumo_port = sumo_port
        sumo_cmd = [
            os.path.join(SUMO_PATH, "bin", sumo_binary),
            f"--remote-port={sumo_port}",
            *base_params,
        ]

        self._log.debug("Starting sumo process:\n\t %s", sumo_cmd)
        self._sumo_proc = subprocess.Popen(
            sumo_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )

    def __del__(self) -> None:
        # We should not raise in delete.
        try:
            self.close_traci_and_pipes(wait=False)
        except Exception:
            pass

    def connect(
        self,
        timeout: float,
        minimum_traci_version: int,
        minimum_sumo_version: Tuple[int, ...],
        debug: bool = False,
    ):
        """Attempt a connection with the SUMO process."""
        traci_conn = None
        try:
            # See if the process is still alive before attempting a connection.
            if self._sumo_proc.poll() is not None:
                raise traci.exceptions.TraCIException(
                    "TraCI server already finished before connection!!!"
                )

            with suppress_output(stderr=not debug, stdout=True):
                traci_conn = traci.connect(
                    self._sumo_port,
                    host=self._host,
                    numRetries=max(0, int(20 * timeout)),
                    proc=self._sumo_proc,
                    waitBetweenRetries=0.05,
                )  # SUMO must be ready within timeout seconds
        # We will retry since this is our first sumo command
        except traci.exceptions.FatalTraCIError as err:
            self._log.error(
                "[%s] TraCI could not connect in time to '%s:%s' [%s]",
                self._name,
                self._host,
                self._sumo_port,
                err,
            )
            # XXX: Actually not fatal...
            raise
        except traci.exceptions.TraCIException as err:
            self._log.error(
                "[%s] SUMO process died while trying to connect to '%s:%s' [%s]",
                self._name,
                self._host,
                self._sumo_port,
                err,
            )
            self.close_traci_and_pipes()
            raise
        except ConnectionRefusedError:
            self._log.error(
                "[%s] Intended TraCI server '%s:%s' refused connection.",
                self._name,
                self._host,
                self._sumo_port,
            )
            self.close_traci_and_pipes()
            raise
        self._connected = True
        self._traci_conn = traci_conn
        try:
            vers, vers_str = traci_conn.getVersion()
            if vers < minimum_traci_version:
                raise ValueError(
                    f"TraCI API version must be >= {minimum_traci_version}. Got version ({vers})"
                )
            self._sumo_version = tuple(
                int(v) for v in vers_str.partition(" ")[2].split(".")
            )  # e.g. "SUMO 1.11.0" -> (1, 11, 0)
            if self._sumo_version < minimum_sumo_version:
                raise ValueError(f"SUMO version must be >= SUMO {minimum_sumo_version}")
        except (traci.exceptions.FatalTraCIError) as err:
            self._log.error(
                "[%s] TraCI disconnected for connection attempt '%s:%s': [%s]",
                self._name,
                self._host,
                self._sumo_port,
                err,
            )
            # XXX: the error type is changed to TraCIException to make it consistent with the
            # process died case of `traci.connect`. Since TraCIException is fatal just in this case...
            self.close_traci_and_pipes()
            raise traci.exceptions.TraCIException(err)
        except OSError as err:
            self._log.error(
                "[%s] OS error occurred for TraCI connection attempt '%s:%s': [%s]",
                self._name,
                self._host,
                self._sumo_port,
                err,
            )
            self.close_traci_and_pipes()
            raise traci.exceptions.TraCIException(err)
        except ValueError:
            self.close_traci_and_pipes()
            raise

    @property
    def connected(self) -> bool:
        """Check if the connection is still valid."""
        return self._sumo_proc is not None and self._connected

    @property
    def viable(self) -> bool:
        """If making a connection to the sumo process is still viable."""
        return self._sumo_proc is not None and self._sumo_proc.poll() is None

    @property
    def sumo_version(self) -> Tuple[int, ...]:
        """Get the current SUMO version as a tuple."""
        return self._sumo_version

    @property
    def port(self) -> Optional[int]:
        """Get the used TraCI port."""
        return self._sumo_port

    @property
    def hostname(self) -> str:
        """Get the used TraCI port."""
        return self._host

    def __getattr__(self, name: str) -> Any:
        if not self.connected:
            raise traci.exceptions.FatalTraCIError("TraCI died.")

        attribute = getattr(self._traci_conn, name)

        if inspect.isbuiltin(attribute) or inspect.ismethod(attribute):
            attribute = functools.partial(
                _wrap_traci_method,
                method=attribute,
                attribute_name=name,
                traci_conn=self,
            )
        elif isinstance(attribute, traci.domain.Domain):
            attribute = DomainWrapper(
                traci_conn=self, domain=attribute, attribute_name=name
            )
        else:
            raise NotImplementedError()

        return attribute

    def must_reset(self):
        """If the version of sumo will have errors if just reloading such that it must be reset."""
        return self._sumo_version > (1, 12, 0)

    def close_traci_and_pipes(self, wait: bool = True, kill: bool = True):
        """Safely closes all connections. We should expect this method to always work without throwing"""

        def __safe_close(conn, **kwargs):
            try:
                conn.close(**kwargs)
            except (subprocess.SubprocessError, multiprocessing.ProcessError):
                # Subprocess or process failed
                pass
            except traci.exceptions.FatalTraCIError:
                # TraCI connection is already dead.
                pass
            except AttributeError:
                # Socket was destroyed internally, likely due to an error.
                pass
            except Exception as err:
                self._log.error("Different error occurred: [%s]", err)

        if self._connected:
            self._log.debug("Closing TraCI connection to %s", self._sumo_port)
            __safe_close(self._traci_conn, wait=wait)

        if self._sumo_proc:
            __safe_close(self._sumo_proc.stdin)
            __safe_close(self._sumo_proc.stdout)
            __safe_close(self._sumo_proc.stderr)
            if kill:
                self._sumo_proc.kill()
                self._sumo_proc = None
                self._log.info(
                    "Killed TraCI server process '%s:%s", self._host, self._sumo_port
                )

        self._connected = False

    def teardown(self):
        """Clean up all resources."""
        self.close_traci_and_pipes()


def _wrap_traci_method(
    *args, method, traci_conn: TraciConn, attribute_name: str, **kwargs
):
    # Argument order must be `*args` first so `method` and `sumo_process` are keyword only arguments.
    try:
        return method(*args, **kwargs)
    except traci.exceptions.FatalTraCIError as err:
        logging.error(
            "[%s] TraCI '%s:%s' disconnected for call '%s', process may have died: [%s]",
            traci_conn._name,
            traci_conn.hostname,
            traci_conn.port,
            attribute_name,
            err,
        )
        # TraCI cannot continue
        traci_conn.close_traci_and_pipes()
        raise traci.exceptions.FatalTraCIError("TraCI died.") from err
    except OSError as err:
        logging.error(
            "[%s] OS error occurred for TraCI '%s:%s' call '%s': [%s]",
            traci_conn._name,
            traci_conn.hostname,
            traci_conn.port,
            attribute_name,
            err,
        )
        traci_conn.close_traci_and_pipes()
        raise OSError("Connection dropped.") from err
    except traci.exceptions.TraCIException as err:
        # Case where TraCI/SUMO can theoretically continue
        raise traci.exceptions.TraCIException("TraCI can continue.") from err
    except KeyboardInterrupt:
        traci_conn.close_traci_and_pipes(wait=False)
        raise
