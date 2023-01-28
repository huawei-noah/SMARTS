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
import argparse
import asyncio
import bisect
import importlib.resources as pkg_resources
import json
import logging
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import ijson
import tornado.gen
import tornado.ioloop
import tornado.iostream
import tornado.web
import tornado.websocket
from tornado.websocket import WebSocketClosedError

import smarts.core.models
from envision.web import dist as web_dist
from smarts.core.utils.file import path2hash

logging.basicConfig(level=logging.WARNING)


# Mapping of simulation IDs to a set of web client run loops
WEB_CLIENT_RUN_LOOPS = {}

# Mapping of simulation ID to the Frames data store
FRAMES = {}

# Mapping of path to map geometry files
MAPS = {}


class AllowCORSMixin:
    """A mixin that adds CORS headers to the page."""

    _HAS_DYNAMIC_ATTRIBUTES = True

    def set_default_headers(self):
        """Setup the default headers.
        In this case they are the minimum required CORS releated headers.
        """
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/octet-stream")

    def options(self):
        """Apply a base response status."""
        self.set_status(204)
        self.finish()


class Frame:
    """A frame that describes a single envision simulation step."""

    def __init__(self, data: str, timestamp: float, next_=None):
        """data is a State object that was converted to string using json.dumps"""
        self._timestamp = timestamp
        self._data = data
        self._size = sys.getsizeof(data)
        self.next_ = next_

    @property
    def timestamp(self):
        """The timestamp for this frame."""
        return self._timestamp

    @property
    def data(self):
        """The raw envision data."""
        return self._data

    @data.setter
    def data(self, data: str):
        self._data = data
        self._size = sys.getsizeof(data)

    @property
    def size(self):
        """The byte size of the frame's raw data."""
        return self._size


class Frames:
    """A managed collection of simulation frames.
    This collection uses a random discard of simulation frames to stay under capacity.
    Random discard favours preserving newer frames over time.
    """

    def __init__(self, max_capacity_mb=500):
        self._max_capacity = max_capacity_mb

        # XXX: Index of timestamp in `self._timestamps` matches the index of the
        # frame with the same timestamp in `self._frames`.
        self._timestamps: List[float] = []
        self._frames: List[Frame] = []

    @property
    def start_frame(self) -> Optional[Frame]:
        """The first frame in all available frames."""
        return self._frames[0] if self._frames else None

    @property
    def start_time(self) -> Optional[float]:
        """The first timestamp in all available frames."""
        return self._frames[0].timestamp if self._frames else None

    @property
    def elapsed_time(self) -> float:
        """The total elapsed time between the first and last frame."""
        if len(self._frames) == 0:
            return 0.0
        return self._frames[-1].timestamp - self._frames[0].timestamp

    def append(self, frame: Frame):
        """Add a frame to the end of the existing frames."""
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

    def __init__(
        self,
        frames: Frames,
        web_client_handler: tornado.websocket.WebSocketHandler,
        message_frequency: float,
        message_frame_volume: int,
        seek: Optional[int] = None,
    ):
        self._log = logging.getLogger(__class__.__name__)
        self._frames: Frames = frames
        self._client: tornado.websocket.WebSocketHandler = web_client_handler
        self._message_wait_time: float = 1 / (max(0, message_frequency) or math.inf)
        self._message_frame_volume: int = message_frame_volume
        self._seek: Optional[int] = seek
        self._thread: Optional[threading.Thread] = None

    def seek(self, offset_seconds):
        """Indicate to the webclient that it should progress to the nearest frame to the given time."""
        self._seek = offset_seconds

    def stop(self):
        """End the simulation playback."""
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def run_forever(self):
        """Starts the connection loop to push frames to the connected web client."""

        async def run_loop():
            frame_ptr = None
            frames_to_send = []

            # wait until we have a start_frame...
            self._log.debug("Waiting for first frame.")
            while frame_ptr is None:
                time.sleep(self._message_wait_time)
                frame_ptr = self._frames.start_frame
                frames_to_send = [frame_ptr]
            self._log.debug("First frame ready.")

            while True:
                # Handle seek
                if self._seek is not None and self._frames.start_time is not None:
                    frame_ptr = self._frames(self._frames.start_time + self._seek)
                    if not frame_ptr:
                        self._log.warning(
                            "Seek frame missing, reverting to start frame"
                        )
                        frame_ptr = self._frames.start_frame
                    frames_to_send.append(frame_ptr)
                    self._seek = None

                # Try to get next frames
                while (
                    frame_ptr.next_
                    and len(frames_to_send) <= self._message_frame_volume
                ):
                    frame_ptr = frame_ptr.next_
                    frames_to_send.append(frame_ptr)

                # If we have new frames, send them to the web client
                if len(frames_to_send) > 0:
                    closed = self._push_frames_to_web_client(frames_to_send)
                    if closed:
                        self._log.debug("Socket closed, exiting")
                        return
                # Otherwise, sleep until next polling loop
                else:
                    delay = self._calculate_frame_delay(frame_ptr)
                    time.sleep(delay)

                # Reset frame list
                frames_to_send = []

        def sync_run_forever():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(run_loop())
            loop.close()

        self._thread = threading.Thread(target=sync_run_forever, args=(), daemon=True)
        self._thread.start()

    def _push_frames_to_web_client(self, frames: List[Frame]):
        try:
            frames_formatted = [
                {
                    "state": frame.data,
                    "current_elapsed_time": frame.timestamp - self._frames.start_time,
                    "total_elapsed_time": self._frames.elapsed_time,
                }
                for frame in frames
            ]
            self._client.write_message(json.dumps(frames_formatted))
            return False
        except WebSocketClosedError:
            self._log.debug("Web client push loop ended.")
            return True

    def _calculate_frame_delay(self, frame_ptr):
        # we may want to be more clever here in the future...
        return self._message_wait_time


class BroadcastWebSocket(tornado.websocket.WebSocketHandler):
    """This websocket receives the SMARTS state (the other end of the open websocket
    is held by the Envision Client (SMARTS)) and broadcasts it to all web clients
    that have open websockets via the `StateWebSocket` handler.
    """

    def initialize(self, max_capacity_mb):
        """Setup this websocket."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._max_capacity_mb = max_capacity_mb

    async def open(self, simulation_id):
        """Asynchronously open the websocket to broadcast to all web clients."""
        self._logger.debug(f"Broadcast websocket opened for simulation={simulation_id}")
        self._simulation_id = simulation_id
        self._frames = Frames(max_capacity_mb=self._max_capacity_mb)
        FRAMES[simulation_id] = self._frames
        WEB_CLIENT_RUN_LOOPS[simulation_id] = set()

    def on_close(self):
        """Close the broadcast websocket."""
        self._logger.debug(
            f"Broadcast websocket closed for simulation={self._simulation_id}"
        )
        del WEB_CLIENT_RUN_LOOPS[self._simulation_id]
        del FRAMES[self._simulation_id]

    async def on_message(self, message):
        """Asynchronously receive messages from the Envision client."""
        it = ijson.parse(message)
        next(it)  # Discard first entry: prefix="", event="start_array", value=None
        prefix, event, value = next(it)
        if prefix == "item" and event == "number":
            # If the second event is a `number`, it is a payload message.
            frame_time = float(value)
            assert isinstance(frame_time, float)
            self._frames.append(Frame(timestamp=frame_time, data=message))
        elif prefix == "item" and event == "start_map":
            # If the second event is a `start_map`, it is a preamble.
            scenarios = [
                value
                for prefix, event, value in it
                if prefix == "item.scenarios.item" and event == "string"
            ]
            path_map = _index_map(scenarios)
            MAPS.update(path_map)
        else:
            raise tornado.web.HTTPError(400, f"Bad request message.")


class StateWebSocket(tornado.websocket.WebSocketHandler):
    """This websocket sits on the other side of the web client. It handles playback and playback
    control messages from the webclient.
    """

    def initialize(self):
        """Setup this websocket."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def check_origin(self, origin):
        """Check the validity of the message origin."""
        return True

    def get_compression_options(self):
        """Get the message compression configuration."""
        return {"compression_level": 6, "mem_level": 5}

    async def open(self, simulation_id):
        """Open this socket to listen for webclient playback requests."""
        if simulation_id not in WEB_CLIENT_RUN_LOOPS:
            raise tornado.web.HTTPError(404, f"Simulation `{simulation_id}` not found.")

        frequency = 10
        message_frame_volume = 100
        self._run_loop = WebClientRunLoop(
            frames=FRAMES[simulation_id],
            web_client_handler=self,
            message_frequency=frequency,
            message_frame_volume=message_frame_volume,
            seek=self.get_argument("seek", None),
        )

        self._logger.debug(f"State websocket opened for simulation={simulation_id}")
        WEB_CLIENT_RUN_LOOPS[simulation_id].add(self._run_loop)

        self._run_loop.run_forever()

    def on_close(self):
        """Stop listening and close the socket."""
        self._logger.debug(f"State websocket closed")
        for run_loop in WEB_CLIENT_RUN_LOOPS.values():
            if self in run_loop:
                self._run_loop.stop()
                run_loop.remove(self._run_loop)

    async def on_message(self, message):
        """Asynchonously handle playback requests."""
        message = json.loads(message)
        if "seek" in message:
            self._run_loop.seek(message["seek"])


class FileHandler(AllowCORSMixin, tornado.web.RequestHandler):
    """This handler serves files to the given requestee."""

    def initialize(self, path_map: Dict[str, Union[str, Path]] = {}):
        """FileHandler that serves file for a given ID."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._path_map = path_map

    async def get(self, id_):
        """Serve a resource requested by id."""
        if id_ not in self._path_map:
            raise tornado.web.HTTPError(404, f"Map resource {id_} not found")

        if not Path(self._path_map[id_]).exists():
            raise tornado.web.HTTPError(
                404, f"Map file `{self._path_map[id_]}` not found."
            )

        await self.serve_chunked(Path(self._path_map[id_]))

    async def serve_chunked(self, path: Path, chunk_size: int = 1024 * 1024):
        """Serve a file to the endpoint given a path."""
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
    """This handler serves map geometry to the given endpoint."""

    def initialize(self):
        """Setup this handler."""
        super().initialize(path_map=MAPS)


class SimulationListHandler(AllowCORSMixin, tornado.web.RequestHandler):
    """This handler serves a list of the active simulations."""

    async def get(self):
        """Serve the active simulations."""
        response = json.dumps({"simulations": list(WEB_CLIENT_RUN_LOOPS.keys())})
        self.write(response)


class ModelFileHandler(FileHandler):
    """This model file handler serves vehicle and other models to the client."""

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
                "motorcycle.glb": "motorcycle.glb",
                "pedestrian.glb": "pedestrian.glb",
            }
        )

    async def get(self, id_):
        """Serve the requested model geometry."""
        if id_ not in self._path_map or not pkg_resources.is_resource(
            smarts.core.models, self._path_map[id_]
        ):
            raise tornado.web.HTTPError(404, f"GLB Model `{id_}` not found.")

        with pkg_resources.path(smarts.core.models, self._path_map[id_]) as path:
            await self.serve_chunked(path)


class MainHandler(tornado.web.RequestHandler):
    """This handler serves the index file."""

    def get(self):
        """Serve the index file."""
        with pkg_resources.path(web_dist, "index.html") as index_path:
            self.render(str(index_path))


def make_app(max_capacity_mb: float, debug: bool):
    """Create the envision web server application through composition of services."""

    dist_path = Path(os.path.dirname(web_dist.__file__)).absolute()
    logging.debug("Creating app with resources at: `%s`", dist_path)
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
            ),
            (r"/assets/models/(.*)", ModelFileHandler),
            (r"/(.*)", tornado.web.StaticFileHandler, dict(path=str(dist_path))),
        ],
        debug=debug,
    )


def _index_map(scenario_dirs: Sequence[str]) -> Dict[str, Path]:
    """Finds and indexes all map geometry files in the given scenario directories."""
    path_map = {}
    for dir_ in scenario_dirs:
        path_map.update(
            {
                f"{path2hash(str(glb.parents[2].resolve()))}.glb": glb
                for glb in Path(dir_).rglob("build/map/map.glb")
            }
        )

    return path_map


def on_shutdown():
    """Callback on shutdown of the envision server."""
    logging.debug("Shutting down Envision")
    tornado.ioloop.IOLoop.current().stop()


def run(
    max_capacity_mb: int = 500,
    port: int = 8081,
    debug: bool = False,
):
    """Create and run an envision web server."""
    app = make_app(max_capacity_mb, debug=debug)
    app.listen(port)
    logging.debug("Envision listening on port=%s", port)

    ioloop = tornado.ioloop.IOLoop.current()
    signal.signal(
        signal.SIGINT, lambda signal, _: ioloop.add_callback_from_signal(on_shutdown)
    )
    signal.signal(
        signal.SIGTERM, lambda signal, _: ioloop.add_callback_from_signal(on_shutdown)
    )
    ioloop.start()


def main():
    """Main function for when using this file as an entrypoint."""
    parser = argparse.ArgumentParser(
        prog="Envision Server",
        description="The Envision server broadcasts SMARTS state to Envision web "
        "clients for visualization.",
    )
    parser.add_argument(
        "--port",
        help="Port Envision will run on.",
        default=8081,
        type=int,
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
    parser.add_argument(
        "--debug", help="Run the server with debug mode.", action="store_true"
    )
    args = parser.parse_args()

    run(
        max_capacity_mb=args.max_capacity,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
