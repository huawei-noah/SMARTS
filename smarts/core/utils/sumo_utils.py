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

import abc
import functools
import inspect
import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
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


def _safe_close(conn, **kwargs):
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
        pass


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


class SumoProcess(metaclass=abc.ABCMeta):
    """A simplified utility representing a SUMO process."""

    @abc.abstractmethod
    def generate(
        self, base_params: List[str], sumo_binary: Literal["sumo", "sumo-gui"] = "sumo"
    ):
        """Generate the process."""
        raise NotImplementedError

    @abc.abstractmethod
    def terminate(self, kill: bool):
        """Terminate this process."""
        raise NotImplementedError

    @abc.abstractmethod
    def poll(self) -> Optional[int]:
        """Poll the underlying process."""
        raise NotImplementedError

    @abc.abstractmethod
    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait on the underlying process."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def port(self) -> int:
        """The port this process is associated with."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def host(self) -> str:
        """The port this process is associated with."""
        raise NotImplementedError


class RemoteSumoProcess(SumoProcess):
    """Connects to a sumo server."""

    def __init__(self, remote_host, remote_port) -> None:
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._port = None
        self._host = None
        self._client_socket = None

    def generate(
        self, base_params: List[str], sumo_binary: Literal["sumo", "sumo-gui"] = "sumo"
    ):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Wait on server to start if it needs to.
        error = None
        for _ in range(5):
            try:
                client_socket.connect((self._remote_host, self._remote_port))
            except OSError as err:
                time.sleep(1)
                error = err
                continue
            break
        else:
            raise OSError(
                f"Unable to connect to server {self._remote_host}:{self._remote_port}. Try running again or running the server using `python -m smarts.core.utils.sumo_server`."
            ) from error

        client_socket.send(f"{sumo_binary}:{json.dumps(base_params)}\n".encode("utf-8"))

        self._client_socket = client_socket

        response = client_socket.recv(1024)
        self._host, _, port = response.decode("utf-8").partition(":")
        self._port = int(port)

    def terminate(self, kill: bool):
        self._client_socket.send("e:".encode("utf-8"))
        self._client_socket.close()

    @property
    def port(self) -> int:
        return self._port or 0

    @property
    def host(self) -> str:
        return self._host or "-1"

    def poll(self) -> Optional[int]:
        return None

    def wait(self, timeout: Optional[float] = None) -> int:
        return 0


class LocalSumoProcess(SumoProcess):
    """Connects to a local sumo process."""

    def __init__(self, sumo_port) -> None:
        self._sumo_proc = None
        self._sumo_port = sumo_port

    def generate(
        self, base_params: List[str], sumo_binary: Literal["sumo", "sumo-gui"] = "sumo"
    ):
        if self._sumo_port is None:
            self._sumo_port = networking.find_free_port()
        sumo_cmd = [
            os.path.join(SUMO_PATH, "bin", sumo_binary),
            f"--remote-port={self._sumo_port}",
            *base_params,
        ]
        self._sumo_proc = subprocess.Popen(
            sumo_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    @property
    def port(self) -> int:
        assert self._sumo_port is not None
        return self._sumo_port

    @property
    def host(self) -> str:
        return "localhost"

    def terminate(self, kill):
        if self._sumo_proc:
            _safe_close(self._sumo_proc.stdin)
            _safe_close(self._sumo_proc.stdout)
            _safe_close(self._sumo_proc.stderr)
        if kill:
            self._sumo_proc.kill()
            self._sumo_proc = None

    def poll(self) -> Optional[int]:
        return self._sumo_proc.poll()

    def wait(self, timeout=None):
        return self._sumo_proc.wait(timeout=timeout)


class TraciConn:
    """A simplified utility for connecting to a SUMO process."""

    def __init__(
        self,
        sumo_process: SumoProcess,
        host: str = "localhost",
        name: str = "",
    ):
        self._traci_conn = None
        self._sumo_port = None
        self._sumo_version: Tuple[int, ...] = tuple()
        self._host = host
        self._name = name
        self._log = logging.Logger(self.__class__.__name__)
        self._log = logging
        self._connected = False

        self._sumo_process = sumo_process

    def __del__(self) -> None:
        # We should not raise in delete.
        try:
            self.close_traci_and_pipes()
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
        self._host = self._sumo_process.host
        self._sumo_port = self._sumo_process.port
        try:
            # See if the process is still alive before attempting a connection.
            with suppress_output(stderr=not debug, stdout=True):
                traci_conn = traci.connect(
                    self._sumo_process.port,
                    host=self._sumo_process.host,
                    numRetries=max(0, int(20 * timeout)),
                    proc=self._sumo_process,
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
            if not self.viable:
                raise traci.exceptions.TraCIException("TraCI server already finished!?")
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
        except traci.exceptions.FatalTraCIError as err:
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
        return self._sumo_process is not None and self._connected

    @property
    def viable(self) -> bool:
        """If making a connection to the sumo process is still viable."""
        return self._sumo_process is not None and self._sumo_process.poll() is None

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

        if self._connected:
            self._log.debug("Closing TraCI connection to %s", self._sumo_port)
            _safe_close(self._traci_conn, wait=wait)

        if self._sumo_process:
            self._sumo_process.terminate(kill=kill)
            self._log.info(
                "Killed TraCI server process '%s:%s'", self._host, self._sumo_port
            )
            self._sumo_process = None

        self._connected = False

    def teardown(self):
        """Clean up all resources."""
        self.close_traci_and_pipes(True)
        self._traci_conn = None


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
