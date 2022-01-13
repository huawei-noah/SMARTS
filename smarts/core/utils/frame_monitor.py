# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import time


class FramerateException(Exception):
    """An exception raised if frame rate has increased higher than the desired threshold."""

    def __str__(self):
        return "The monitor has not started yet."

    @classmethod
    def below_threshold(cls, desired_fps, delta):
        """Generate a framerate exception.
        Args:
            desired_fps: The intended fps.
            delta: The frame time taken.
        Returns:
            A new framerate exception.
        """
        return cls(
            f"The frame rate decreased, lower than the desired threshold, \
                desired: {desired_fps} fps, actual: {round(1000 / delta, 2)} fps."
        )


class FrameMonitor:
    """A tool for requiring a minimum frame rate."""

    def __init__(self, desired_fps=10):
        self._desired_fps = int(desired_fps)
        self._maximum_frame_time_ms = round(1 / self._desired_fps, 3) * 1000
        self._start_time_ms = None

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def _time_now(self):
        return int(time.time() * 1000)

    def start(self):
        """Starts timing the frame."""
        self._start_time_ms = self._time_now()

    def stop(self):
        """Ends timing the frame. Throws an exception if the frame duration is greater than
        desired fps.
        """
        if self._start_time_ms is None:
            print("The monitor has not started yet.")
            return -1

        now = self._time_now()
        delta = now - self._start_time_ms
        actual_fps = round(1000 / delta, 2)
        if actual_fps < self._desired_fps:
            raise FramerateException.below_threshold(self._desired_fps, delta)
        return actual_fps
