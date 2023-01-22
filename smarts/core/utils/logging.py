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
import ctypes
import logging
import os
import sys
import warnings
from contextlib import contextmanager
from io import UnsupportedOperation
from time import time


@contextmanager
def timeit(name: str, log):
    """Context manger that stopwatches the amount of time between context block start and end.

    .. code-block:: python

       import logging
       with timeit(n,logging.log):
            a = a * b

    """
    start = time()
    yield
    elapsed_time = (time() - start) * 1000

    log(f'"{name}" took: {elapsed_time:4f}ms')


def isnotebook():
    """Determines if executing in ipython (Jupyter Notebook)"""
    try:
        shell = get_ipython().__class__.__name__  # pytype: disable=name-error
        if shell == "ZMQInteractiveShell" or "google.colab" in sys.modules:
            return True  # Jupyter notebook or qtconsole or Google Colab
    except NameError:
        pass

    return False


libc = ctypes.CDLL(None)
try:
    c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")
    c_stdout = ctypes.c_void_p.in_dll(libc, "stdout")
except:
    # macOS
    c_stderr = ctypes.c_void_p.in_dll(libc, "__stderrp")
    c_stdout = ctypes.c_void_p.in_dll(libc, "__stdoutp")


def try_fsync(fd):
    """Attempts to see if fsync will work. Workaround for error on Github Actions."""
    try:
        os.fsync(fd)
    except OSError:
        # On GH actions was returning an OSError: [Errno 22] Invalid argument
        pass


@contextmanager
def suppress_output(stderr=True, stdout=True):
    """Attempts to suppress console print statements.
    Args:
        stderr: Suppress stderr.
        stdout: Suppress stdout.
    """
    cleanup_stderr = None
    cleanup_stdout = None
    try:
        if stderr:
            cleanup_stderr = _suppress_fileout("stderr")
        if stdout:
            cleanup_stdout = _suppress_fileout("stdout")
        yield
    finally:
        if stderr and cleanup_stderr:
            cleanup_stderr(c_stderr)
        if stdout and cleanup_stdout:
            cleanup_stdout(c_stdout)


def _suppress_fileout(stdname):
    original = getattr(sys, stdname)
    try:
        original_std_fno = original.fileno()
    except UnsupportedOperation as e:
        if not isnotebook():
            raise e
        file = open(os.devnull, "w")
        old_std = getattr(sys, stdname)
        setattr(sys, stdname, file)

        def cleanup_notebook(_):
            nonlocal old_std, stdname
            new_std = getattr(sys, stdname)
            new_std.flush()
            # Ensure attributes exist because of https://github.com/ipython/ipykernel/issues/867
            if not hasattr(new_std, "watch_fd_thread"):
                setattr(new_std, "watch_fd_thread", None)
            if not hasattr(new_std, "_exc"):
                setattr(new_std, "_exc", None)
            new_std.close()
            setattr(sys, stdname, old_std)

        ## This case is notebook
        return cleanup_notebook

    dup_std_fno = os.dup(original_std_fno)
    devnull_fno = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fno, original_std_fno)
    setattr(sys, stdname, os.fdopen(devnull_fno, "w"))

    def cleanup_local(c_stdobj):
        getattr(sys, stdname).flush()
        libc.fflush(c_stdobj)
        try_fsync(devnull_fno)
        os.close(devnull_fno)
        os.dup2(dup_std_fno, original_std_fno)
        os.close(dup_std_fno)
        try:
            getattr(sys, stdname).close()
        except OSError as e:
            # This happens in some environments and is fine so we should ignore just it
            if e.errno != 9:  # [Errno 9] Bad file descriptor
                raise e
        finally:
            setattr(sys, stdname, original)

    return cleanup_local


@contextmanager
def suppress_websocket():
    """Attempts to filter out irritating `websocket` library messages."""

    websocket_filter = lambda record: "goodbye" not in record.msg
    with warnings.catch_warnings():
        # XXX: websocket-client library seems to have leaks on connection
        #      retry that cause annoying warnings within Python 3.8+
        warnings.filterwarnings("ignore", category=ResourceWarning)
        # Filter out the websocket "goodbye" messages.
        _logger = logging.getLogger("websocket")
        _logger.addFilter(websocket_filter)
        yield
        _logger.removeFilter(websocket_filter)
