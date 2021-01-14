import time

import pytest

from smarts.core.utils.frame_rate import FrameMonitor


def func_with_runtime_error():
    with FrameMonitor(25):
        # Forcing execute time longer than 0.04
        time.sleep(0.1)


def func():
    with FrameMonitor(25):
        return ""


def test_frame_rate_monitor():
    func()

    with pytest.raises(RuntimeError):
        func_with_runtime_error()
