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
import json
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

import smarts.core.models
from smarts.core.utils.file import path2hash
from .web import dist as web_dist

logging.basicConfig(level=logging.WARNING)


# Mapping of simulation IDs to a set of web socket handlers
STATE_CLIENTS = {}


class AllowCORSMixin:
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Content-Type", "application/octet-stream")

    def options(self):
        self.set_status(204)
        self.finish()


class BroadcastWebSocket(tornado.websocket.WebSocketHandler):
    """This websocket receives the SMARTS state (the other end of the open websocket
    is held by the Envision Client (SMARTS)) and broadcasts it to all web clients
    that have open websockets via the `StateWebSocket` handler.
    """

    def initialize(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._simulation_id = None

    async def open(self, simulation_id):
        self._logger.debug(f"Broadcast websocket opened for simulation={simulation_id}")
        self._simulation_id = simulation_id
        STATE_CLIENTS[simulation_id] = set()

    def on_close(self):
        self._logger.debug(
            f"Broadcast websocket closed for simulation={self._simulation_id}"
        )
        del STATE_CLIENTS[self._simulation_id]

    async def on_message(self, message):
        for client in STATE_CLIENTS[self._simulation_id]:
            client.write_message(message)


class StateWebSocket(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def check_origin(self, origin):
        return True

    async def open(self, simulation_id):
        if simulation_id not in STATE_CLIENTS:
            raise tornado.web.HTTPError(404)

        self._logger.debug(f"State websocket opened for simulation={simulation_id}")
        STATE_CLIENTS[simulation_id].add(self)

    def on_close(self):
        self._logger.debug(f"State websocket closed")
        for simulation_id, clients in STATE_CLIENTS.items():
            if self in clients:
                clients.remove(self)


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
        response = json.dumps({"simulations": list(STATE_CLIENTS.keys())})
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


def make_app(scenario_dirs: Sequence):
    with pkg_resources.path(web_dist, ".") as dist_path:
        return tornado.web.Application(
            [
                (r"/", MainHandler),
                (r"/simulations", SimulationListHandler),
                (r"/simulations/(?P<simulation_id>\w+)/state", StateWebSocket),
                (r"/simulations/(?P<simulation_id>\w+)/broadcast", BroadcastWebSocket),
                (
                    r"/assets/maps/(.*)",
                    MapFileHandler,
                    dict(scenario_dirs=scenario_dirs),
                ),
                (r"/assets/models/(.*)", ModelFileHandler),
                (r"/(.*)", tornado.web.StaticFileHandler, {"path": str(dist_path)}),
            ]
        )


def on_shutdown():
    logging.debug("Shutting down Envision")
    tornado.ioloop.IOLoop.current().stop()


def run(scenario_dirs, port=8081):
    app = make_app(scenario_dirs)
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
    args = parser.parse_args()

    run(scenario_dirs=args.scenarios, port=args.port)


if __name__ == "__main__":
    main()
