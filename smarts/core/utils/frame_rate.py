import time


class FrameMonitor:
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
        self._start_time_ms = self._time_now()

    def stop(self):
        if self._start_time_ms is None:
            raise RuntimeError("The monitor has not started yet.")

        now = self._time_now()
        delta = now - self._start_time_ms
        if delta > self._maximum_frame_time_ms:
            raise RuntimeError(
                f"The frame rate drops, lower than the desired threshold, \
                desired: {self._desired_fps} fps, actual: {round(1000 / delta, 2)} fps."
            )
