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
import logging
import os
import random
import tempfile

import sh
from yattag import Doc, indent

from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.core.utils.file import make_dir_in_smarts_log_dir
from smarts.core.utils.sumo import sumolib

from . import types


class InvalidRoute(Exception):
    """An exception given if a route cannot be successfully plotted."""

    pass


class RandomRouteGenerator:
    """Generates a random route out of the routes available in the road network.

    Args:
        road_network:
            A network of routes defined for vehicles of different kinds to travel on.
    """

    def __init__(self, road_network: SumoRoadNetwork):
        self._log = logging.getLogger(self.__class__.__name__)
        self._road_network = road_network

    @classmethod
    def from_file(cls, net_file: str):
        """Constructs a route generator from the given file

        Args:
            net_file: The path to a '\\*.net.xml' file (generally 'map.net.xml')
        """
        # XXX: Spacing is crudely "large enough" so we less likely overlap vehicles
        road_network = SumoRoadNetwork.from_file(net_file, lanepoint_spacing=2.0)
        return cls(road_network)

    def __iter__(self):
        return self

    def __next__(self):
        """Provides the next random route."""

        def random_lane_index(edge_id):
            lanes = self._road_network.edge_by_id(edge_id).getLanes()
            return random.randint(0, len(lanes) - 1)

        def random_lane_offset(edge_id, lane_idx):
            lane = self._road_network.edge_by_id(edge_id).getLanes()[lane_idx]
            return random.uniform(0, lane.getLength())

        # HACK: loop + continue is a temporary solution so we more likely return a valid
        #       route. In future we need to be able to handle random routes that are just
        #       a single edge long.
        for _ in range(100):
            edges = self._road_network.random_route(max_route_len=10)
            if len(edges) < 2:
                continue

            start_edge_id = edges[0]
            start_lane_index = random_lane_index(start_edge_id)
            start_lane_offset = random_lane_offset(start_edge_id, start_lane_index)

            end_edge_id = edges[-1]
            end_lane_index = random_lane_index(end_edge_id)
            end_lane_offset = random_lane_offset(end_edge_id, end_lane_index)

            return types.Route(
                begin=(start_edge_id, start_lane_index, start_lane_offset),
                via=tuple(edges[1:-1]),
                end=(end_edge_id, end_lane_index, end_lane_offset),
            )

        raise InvalidRoute(
            "Unable to generate a valid random route that contains \
            at least two edges."
        )


class TrafficGenerator:
    def __init__(
        self,
        scenario_dir: str,
        log_dir: str = None,
        overwrite: bool = False,
    ):
        """
        scenario: The path to the scenario directory
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self._scenario = scenario_dir
        self._overwrite = overwrite
        self._duarouter = sh.Command(sumolib.checkBinary("duarouter"))
        self._road_network_path = os.path.join(self._scenario, "map.net.xml")
        self._road_network = None
        self._random_route_generator = None
        self._log_dir = self._resolve_log_dir(log_dir)

    def plan_and_save(
        self, traffic: types.Traffic, name: str, output_dir: str = None, seed: int = 42
    ):
        """Writes a traffic spec to a route file in the given output_dir.

        traffic: The traffic to write out
        name: The name of the traffic resource file
        output_dir: The output directory of the traffic file
        seed: The seed used for deterministic random calls
        """
        random.seed(seed)

        route_path = os.path.join(output_dir, "{}.rou.xml".format(name))
        route_alt_path = os.path.join(output_dir, "{}.rou.alt.xml".format(name))

        if os.path.exists(route_path):
            if self._overwrite:
                self._log.info(
                    f"Routes at routes={route_path} already exist, overwriting"
                )
            else:
                self._log.info(f"Routes at routes={route_path} already exist, skipping")
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            trips_path = os.path.join(temp_dir, "trips.trips.xml")
            self._writexml(traffic, trips_path)

            scenario_name = os.path.basename(os.path.normpath(output_dir))
            log_path = f"{self._log_dir}/{scenario_name}"
            os.makedirs(log_path, exist_ok=True)

            # Validates, and runs route planner
            self._duarouter(
                unsorted_input=True,
                net_file=self.road_network.net_file,
                route_files=trips_path,
                output_file=route_path,
                seed=seed,
                ignore_errors=False,
                no_step_log=True,
                repair=True,
                error_log=f"{log_path}/{name}.log",
            )

            # Remove the rou.alt.xml file
            if os.path.exists(route_alt_path):
                os.remove(route_alt_path)

        return route_path

    def _writexml(self, traffic: types.Traffic, route_path: str):
        """Writes a traffic spec into a route file. Typically this would be the source
        data to Sumo's DUAROUTER.
        """
        doc = Doc()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        with doc.tag(
            "routes",
            ("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"),
            ("xsi:noNamespaceSchemaLocation", "http://sumo.sf.net/xsd/routes_file.xsd"),
        ):
            # Actors and routes may be declared once then reused. To prevent creating
            # duplicates we unique them here.
            for actor in {
                actor for flow in traffic.flows for actor in flow.actors.keys()
            }:
                sigma = min(1, max(0, actor.imperfection.sample()))  # range [0,1]
                min_gap = max(0, actor.min_gap.sample())  # range >= 0
                doc.stag(
                    "vType",
                    id=actor.id,
                    accel=actor.accel,
                    decel=actor.decel,
                    vClass=actor.vehicle_type,
                    speedFactor=actor.speed.mean,
                    speedDev=actor.speed.sigma,
                    sigma=sigma,
                    minGap=min_gap,
                    maxSpeed=actor.max_speed,
                    **actor.lane_changing_model,
                    **actor.junction_model,
                )

            # Make sure all routes are "resolved" (e.g. `RandomRoute` are converted to
            # `Route`) so that we can write them all to file.
            resolved_routes = {}
            for route in {flow.route for flow in traffic.flows}:
                resolved_routes[route] = self.resolve_route(route)

            for route in set(resolved_routes.values()):
                doc.stag("route", id=route.id, edges=" ".join(route.edges))

            # We don't de-dup flows since defining the same flow multiple times should
            # create multiple traffic flows. Since IDs can't be reused, we also unique
            # them here.
            for flow_idx, flow in enumerate(traffic.flows):
                total_weight = sum(flow.actors.values())
                route = resolved_routes[flow.route]
                for actor_idx, (actor, weight) in enumerate(flow.actors.items()):
                    doc.stag(
                        "flow",
                        id="{}-{}-{}-{}".format(
                            actor.name, flow.id, flow_idx, actor_idx
                        ),
                        type=actor.id,
                        route=route.id,
                        vehsPerHour=flow.rate * (weight / total_weight),
                        departLane=route.begin[1],
                        departPos=route.begin[2],
                        departSpeed=actor.depart_speed,
                        arrivalLane=route.end[1],
                        arrivalPos=route.end[2],
                        begin=flow.begin,
                        end=flow.end,
                    )

        with open(route_path, "w") as f:
            f.write(
                indent(
                    doc.getvalue(), indentation="    ", newline="\r\n", indent_text=True
                )
            )

    def _cache_road_network(self):
        if not self._road_network:
            self._road_network = SumoRoadNetwork.from_file(self._road_network_path)

    def resolve_edge_length(self, edge_id, lane_id):
        self._cache_road_network()
        lane = self._road_network.edge_by_id(edge_id).getLanes()[lane_id]
        return lane.getLength()

    def resolve_route(self, route):
        if not isinstance(route, types.RandomRoute):
            return route

        if not self._random_route_generator:
            # Lazy-load to improve performance when not using random route generation.
            self._random_route_generator = RandomRouteGenerator.from_file(
                self._road_network_path
            )

        return next(self._random_route_generator)

    @property
    def road_network(self):
        self._cache_road_network()
        return self._road_network

    def _resolve_log_dir(self, log_dir):
        if log_dir is None:
            log_dir = make_dir_in_smarts_log_dir("_duarouter_routing")

        return os.path.abspath(log_dir)
