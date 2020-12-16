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
import time
import sys
import random
import asyncio
import threading
import json
import bisect
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Sequence
import importlib.resources as pkg_resources

import tornado.web
import tornado.gen
import tornado.ioloop
import tornado.iostream
import tornado.websocket
from tornado.websocket import WebSocketClosedError

import smarts.core.models
from smarts.core.utils.file import path2hash
from envision.web import dist as web_dist

logging.basicConfig(level=logging.WARNING)


# Mapping of simulation IDs to a set of web client run loops
WEB_CLIENT_RUN_LOOPS = {}

# Mapping of simulation ID to the Frames data store
FRAMES = {}


class AllowCORSMixin:
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/octet-stream")

    def options(self):
        self.set_status(204)
        self.finish()


class Frame:
    def __init__(self, data: str, timestamp: int, next_=None):
        # Time since epoch in seconds
        self._timestamp = timestamp
        self._data = data
        self._size = sys.getsizeof(data)
        self.next_ = next_

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: str):
        self._data = data
        self._size = sys.getsizeof(data)

    @property
    def size(self):
        return self._size


class Frames:
    def __init__(self, max_capacity_mb=500):
        self._max_capacity = max_capacity_mb

        # XXX: Index of timestamp in `self._timestamps` matches the index of the
        # frame with the same timestamp in `self._frames`.
        self._timestamps = []
        self._frames = []

    @property
    def start_frame(self):
        if len(self._frames) == 0:
            return None

        return self._frames[0]

    @property
    def start_time(self):
        return self._frames[0].timestamp if self._frames else None

    @property
    def elapsed_time(self):
        if len(self._frames) == 0:
            return 0

        return self._frames[-1].timestamp - self._frames[0].timestamp

    def append(self, frame: Frame):
        self._enforce_max_capacity()

        if len(self._frames) >= 1:
            self._frames[-1].next_ = frame

        self._frames.append(frame)
        self._timestamps.append(frame.timestamp)

    def __call__(self, timestamp):
        """Finds the nearest frame according to the given timestamp."""
        frame_idx = bisect.bisect_left(self._timestamps, timestamp)
        if frame_idx >= len(self._frames):
            frame_idx = -1

        return self._frames[frame_idx]

    def _enforce_max_capacity(self):
        """Sample random frames and clear their data to ensure we're under the max
        capacity size.
        """
        bytes_to_mb = 1e-6
        start_frames_to_keep = 1
        end_frames_to_keep = 10
        sizes = [frame.size for frame in self._frames]
        while (
            len(self._frames) > start_frames_to_keep + end_frames_to_keep
            and sum(sizes) * bytes_to_mb > self._max_capacity
        ):
            # XXX: randint(1, ...), we skip the start frame because that is a "safe"
            #      frame we can always rely on being available.
            idx_to_delete = random.randint(
                1, len(self._frames) - 1 - end_frames_to_keep
            )
            self._frames[idx_to_delete - 1].next_ = self._frames[idx_to_delete].next_
            del sizes[idx_to_delete]
            del self._frames[idx_to_delete]
            del self._timestamps[idx_to_delete]


class WebClientRunLoop:
    """The run loop is like a "video player" for the simulation. It supports seeking
    and playback. The run loop wraps the web client handler and pushes the frame
    messages to it as needed.
    """

    def __init__(self, frames, web_client_handler, timestep_sec, seek=None):
        self._log = logging.getLogger(__class__.__name__)
        self._frames = frames
        self._client = web_client_handler
        self._timestep_sec = timestep_sec
        self._seek = seek
        self._thread = None

    def seek(self, offset_seconds):
        self._seek = offset_seconds

    def stop(self):
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def run_forever(self):
        async def run_loop():
            # If no frame, wait till one is present
            frame = self._frames.start_frame
            while frame is None:
                await asyncio.sleep(0.1)
                frame = self._frames.start_frame

            while True:
                # Handle seek
                if self._seek is not None and self._frames.start_time is not None:
                    frame = self._frames(self._frames.start_time + self._seek)
                    self._seek = None

                if frame is None:
                    self._log.warning("Seek frame missing, reverting to start frame")
                    frame = self._frames.start_frame

                closed = self._push_frame_to_web_client(frame)
                if closed:
                    self._log.debug("Socket closed, exiting")
                    return

                frame = await self._wait_for_next_frame(frame)

        def sync_run_forever():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(run_loop())
            loop.close()

        self._thread = threading.Thread(target=sync_run_forever, args=(), daemon=True)
        self._thread.start()

    def _push_frame_to_web_client(self, frame):
        try:
            self._client.write_message(
                json.dumps(
                    {
                        "state": frame.data,
                        "current_elapsed_time": frame.timestamp
                        - self._frames.start_time,
                        "total_elapsed_time": self._frames.elapsed_time,
                    }
                )
            )
            return False
        except WebSocketClosedError:
            return True

    async def _wait_for_next_frame(self, frame):
        while True:
            # TODO: Consider using an asyncio queue instead
            await asyncio.sleep(self._timestep_sec)
            if frame.next_ is not None:
                return frame.next_


class BroadcastWebSocket(tornado.websocket.WebSocketHandler):
    """This websocket receives the SMARTS state (the other end of the open websocket
    is held by the Envision Client (SMARTS)) and broadcasts it to all web clients
    that have open websockets via the `StateWebSocket` handler.
    """

    def initialize(self, max_capacity_mb):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_capacity_mb = max_capacity_mb

    async def open(self, simulation_id):
        self._logger.debug(f"Broadcast websocket opened for simulation={simulation_id}")
        self._simulation_id = simulation_id
        self._frames = Frames(max_capacity_mb=self._max_capacity_mb)
        FRAMES[simulation_id] = self._frames
        WEB_CLIENT_RUN_LOOPS[simulation_id] = set()

    def on_close(self):
        self._logger.debug(
            f"Broadcast websocket closed for simulation={self._simulation_id}"
        )
        del WEB_CLIENT_RUN_LOOPS[self._simulation_id]
        del FRAMES[self._simulation_id]

    async def on_message(self, message):
        time_since_epoch = time.time()
        frame = Frame(timestamp=time_since_epoch, data=message)
        self._frames.append(frame)


class StateWebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def check_origin(self, origin):
        return True

    def get_compression_options(self):
        return {"compression_level": 6, "mem_level": 5}

    async def open(self, simulation_id):
        if simulation_id not in WEB_CLIENT_RUN_LOOPS:
            raise tornado.web.HTTPError(404)

        # TODO: Set this appropriately
        timestep_sec = 0.1
        self._run_loop = WebClientRunLoop(
            frames=FRAMES[simulation_id],
            web_client_handler=self,
            timestep_sec=timestep_sec,
            seek=self.get_argument("seek", None),
        )

        self._logger.debug(f"State websocket opened for simulation={simulation_id}")
        WEB_CLIENT_RUN_LOOPS[simulation_id].add(self._run_loop)

        self._run_loop.run_forever()

    def on_close(self):
        self._logger.debug(f"State websocket closed")
        for run_loop in WEB_CLIENT_RUN_LOOPS.values():
            if self in run_loop:
                self._run_loop.stop()
                run_loop.remove(self._run_loop)

    async def on_message(self, message):
        message = json.loads(message)
        if "seek" in message:
            self._run_loop.seek(message["seek"])


class FileHandler(AllowCORSMixin, tornado.web.RequestHandler):
    def initialize(self, path_map: Dict[str, Path] = {}):
        """FileHandler that serves file for a given ID."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path_map = path_map

    async def get(self, id_):
        if id_ not in self._path_map or not self._path_map[id_].exists():
            raise tornado.web.HTTPError(404)

        await self.serve_chunked(self._path_map[id_])

    async def serve_chunked(self, path: Path, chunk_size: int = 1024 * 1024):
        with open(path, mode="rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                try:
                    self.write(chunk)
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    break
                finally:
                    del chunk  # to conserve memory
                    # pause the coroutine so other handlers can run
                    await tornado.gen.sleep(1e-9)  # 1 nanosecond


class MapFileHandler(FileHandler):
    def initialize(self, scenario_dirs: Sequence):
        path_map = {}
        for dir_ in scenario_dirs:
            path_map.update(
                {
                    f"{path2hash(str(glb.parent.resolve()))}.glb": glb
                    for glb in Path(dir_).rglob("*.glb")
                }
            )

        super().initialize(path_map)


class SimulationListHandler(AllowCORSMixin, tornado.web.RequestHandler):
    async def get(self):
        response = json.dumps({"simulations": list(WEB_CLIENT_RUN_LOOPS.keys())})
        self.write(response)


class ModelFileHandler(FileHandler):
    def initialize(self):
        # We store the resource filenames as values in `path_map` and route them
        # through `importlib.resources` for resolution.
        super().initialize(
            {
                "muscle_car_agent.glb": "muscle_car.glb",
                "muscle_car_social_agent.glb": "muscle_car.glb",
                "simple_car.glb": "simple_car.glb",
                "bus.glb": "bus.glb",
                "coach.glb": "coach.glb",
                "truck.glb": "truck.glb",
                "trailer.glb": "trailer.glb",
            }
        )

    async def get(self, id_):
        if id_ not in self._path_map or not pkg_resources.is_resource(
            smarts.core.models, self._path_map[id_]
        ):
            raise tornado.web.HTTPError(404)

        with pkg_resources.path(smarts.core.models, self._path_map[id_]) as path:
            await self.serve_chunked(path)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        with pkg_resources.path(web_dist, "index.html") as index_path:
            self.render(str(index_path))


def make_app(scenario_dirs: Sequence, max_capacity_mb: float):
    with pkg_resources.path(web_dist, ".") as dist_path:
        return tornado.web.Application(
            [
                (r"/", MainHandler),
                (r"/simulations", SimulationListHandler),
                (r"/simulations/(?P<simulation_id>\w+)/state", StateWebSocket),
                (
                    r"/simulations/(?P<simulation_id>\w+)/broadcast",
                    BroadcastWebSocket,
                    dict(max_capacity_mb=max_capacity_mb),
                ),
                (
                    r"/assets/maps/(.*)",
                    MapFileHandler,
                    dict(scenario_dirs=scenario_dirs),
                ),
                (r"/assets/models/(.*)", ModelFileHandler),
                (r"/(.*)", tornado.web.StaticFileHandler, dict(path=str(dist_path))),
            ]
        )


def on_shutdown():
    logging.debug("Shutting down Envision")
    tornado.ioloop.IOLoop.current().stop()


def run(scenario_dirs, max_capacity_mb=500, port=8081):
    app = make_app(scenario_dirs, max_capacity_mb)
    app.listen(port)
    logging.debug(f"Envision listening on port={port}")

    ioloop = tornado.ioloop.IOLoop.current()
    signal.signal(
        signal.SIGINT, lambda signal, _: ioloop.add_callback_from_signal(on_shutdown)
    )
    ioloop.start()


def main():
    parser = argparse.ArgumentParser(
        prog="Envision Server",
        description="The Envision server broadcasts SMARTS state to Envision web "
        "clients for visualization.",
    )
    parser.add_argument(
        "--scenarios",
        help="A list of directories where scenarios are stored.",
        default=["./scenarios"],
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--port", help="Port Envision will run on.", default=8081, type=int,
    )
    parser.add_argument(
        "--max_capacity",
        help=(
            "Max capacity in MB of Envision. The larger the more contiguous history "
            "Envision can store."
        ),
        default=500,
        type=float,
    )
    args = parser.parse_args()

    run(scenario_dirs=args.scenarios, max_capacity_mb=args.max_capacity, port=args.port)


if __name__ == "__main__":
    main()
