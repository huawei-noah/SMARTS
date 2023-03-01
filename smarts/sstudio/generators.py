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
import logging
import os
import random
import tempfile
from typing import Optional

import sh
from yattag import Doc, indent

from smarts.core.road_map import RoadMap
from smarts.core.utils.file import make_dir_in_smarts_log_dir, replace

from . import types

SECONDS_PER_HOUR_INV = 1 / 60 / 60


class InvalidRoute(Exception):
    """An exception given if a route cannot be successfully plotted."""

    pass


class RandomRouteGenerator:
    """Generates a random route out of the routes available in the road map.

    Args:
        road_map:
            A network of routes defined for vehicles of different kinds to travel on.
    """

    def __init__(self, road_map: RoadMap):
        self._log = logging.getLogger(self.__class__.__name__)
        self._road_map = road_map

    def __iter__(self):
        return self

    def __next__(self):
        """Provides the next random route."""

        def random_lane_index(road_id: str) -> int:
            lanes = self._road_map.road_by_id(road_id).lanes
            return random.randint(0, len(lanes) - 1)

        def random_lane_offset(road_id: str, lane_idx: int) -> float:
            lane = self._road_map.road_by_id(road_id).lanes[lane_idx]
            return random.uniform(0, lane.length)

        # HACK: loop + continue is a temporary solution so we more likely return a valid
        #       route. In future we need to be able to handle random routes that are just
        #       a single road long.
        for _ in range(100):
            route = self._road_map.random_route(max_route_len=10)
            if len(route.roads) < 2:
                continue

            start_road_id = route.roads[0].road_id
            start_lane_index = random_lane_index(start_road_id)
            start_lane_offset = random_lane_offset(start_road_id, start_lane_index)

            end_road_id = route.roads[-1].road_id
            end_lane_index = random_lane_index(end_road_id)
            end_lane_offset = random_lane_offset(end_road_id, end_lane_index)

            return types.Route(
                begin=(start_road_id, start_lane_index, start_lane_offset),
                via=tuple(road.road_id for road in route.roads[1:-1]),
                end=(end_road_id, end_lane_index, end_lane_offset),
            )

        raise InvalidRoute(
            "Unable to generate a valid random route that contains \
            at least two roads."
        )


class TrafficGenerator:
    """Generates traffic from scenario information."""

    def __init__(
        self,
        scenario_dir: str,
        scenario_map_spec: Optional[types.MapSpec],
        log_dir: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Args:
            scenario:
                The path to the scenario directory.
            scenario_map_spec:
                The map spec information.
            log_dir:
                Where logging information about traffic planning should be written to.
            overwrite:
                Whether to overwrite existing traffic information.
        """

        self._log = logging.getLogger(self.__class__.__name__)
        self._scenario = scenario_dir
        self._overwrite = overwrite
        self._scenario_map_spec = scenario_map_spec
        self._road_network_path = os.path.join(self._scenario, "map.net.xml")
        if scenario_map_spec and scenario_map_spec.source:
            if os.path.isfile(scenario_map_spec.source):
                self._road_network_path = scenario_map_spec.source
            elif os.path.exists(scenario_map_spec.source):
                self._road_network_path = os.path.join(
                    scenario_map_spec.source, "map.net.xml"
                )
        self._road_network = None
        self._random_route_generator = None
        self._log_dir = self._resolve_log_dir(log_dir)

    def plan_and_save(
        self,
        traffic: types.Traffic,
        name: str,
        output_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Writes a traffic spec to a route file in the given output_dir.

        traffic: The traffic to write out
        name: The name of the traffic resource file
        output_dir: The output directory of the traffic file
        seed: The seed used for deterministic random calls
        """
        random.seed(seed)

        ext = "smarts" if traffic.engine == "SMARTS" else "rou"
        route_path = os.path.join(output_dir, "{}.{}.xml".format(name, ext))
        if os.path.exists(route_path):
            if self._overwrite:
                self._log.info(
                    f"Routes at routes={route_path} already exist, overwriting"
                )
            else:
                self._log.info(f"Routes at routes={route_path} already exist, skipping")
                return None

        if traffic.engine == "SMARTS":
            self._writexml(traffic, True, route_path)
            return route_path

        assert (
            traffic.engine == "SUMO"
        ), f"Unsupported traffic engine specified: {traffic.engine}"

        with tempfile.TemporaryDirectory() as temp_dir:
            trips_path = os.path.join(temp_dir, "trips.trips.xml")
            self._writexml(traffic, False, trips_path)

            route_alt_path = os.path.join(output_dir, "{}.rou.alt.xml".format(name))
            scenario_name = os.path.basename(os.path.normpath(output_dir))
            log_path = f"{self._log_dir}/{scenario_name}"
            os.makedirs(log_path, exist_ok=True)

            import smarts.core.utils.sumo  # Set SUMO_HOME environment variable
            from smarts.core.utils.sumo import sumolib

            duarouter = sh.Command(sumolib.checkBinary("duarouter"))

            # Validates, and runs route planner
            duarouter(
                unsorted_input=True,
                net_file=self.road_network.source,
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

    def _writexml(
        self, traffic: types.Traffic, fill_in_route_gaps: bool, route_path: str
    ):
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
            actors_for_vtypes = {
                actor for flow in traffic.flows for actor in flow.actors.keys()
            }
            if traffic.trips:
                actors_for_vtypes |= {trip.actor for trip in traffic.trips}
                vehicle_id_set = {trip.vehicle_name for trip in traffic.trips}
                vehilce_ids_list = [trip.vehicle_name for trip in traffic.trips]
                if len(vehicle_id_set) != len(vehilce_ids_list):
                    raise ValueError("Repeated single vehicle names is not allowed.")

            for actor in actors_for_vtypes:
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
                resolved_routes[route] = self.resolve_route(route, fill_in_route_gaps)

            for route in set(resolved_routes.values()):
                doc.stag("route", id=route.id, edges=" ".join(route.roads))

            # We don't de-dup flows since defining the same flow multiple times should
            # create multiple traffic flows. Since IDs can't be reused, we also unique
            # them here.
            for flow_idx, flow in enumerate(traffic.flows):
                total_weight = sum(flow.actors.values())
                route = resolved_routes[flow.route]
                for actor_idx, (actor, weight) in enumerate(flow.actors.items()):
                    vehs_per_hour = flow.rate * (weight / total_weight)
                    rate_option = {}
                    if flow.randomly_spaced:
                        vehs_per_sec = vehs_per_hour * SECONDS_PER_HOUR_INV
                        rate_option = dict(probability=vehs_per_sec)
                    else:
                        rate_option = dict(vehsPerHour=vehs_per_hour)
                    doc.stag(
                        "flow",
                        # have to encode the flow.repeat_route within the vehcile id b/c
                        # duarouter complains about any additional xml tags or attributes.
                        id="{}-{}{}-{}-{}".format(
                            actor.name,
                            flow.id,
                            "-endless" if flow.repeat_route else "",
                            flow_idx,
                            actor_idx,
                        ),
                        type=actor.id,
                        route=route.id,
                        departLane=route.begin[1],
                        departPos=route.begin[2],
                        departSpeed=actor.depart_speed,
                        arrivalLane=route.end[1],
                        arrivalPos=route.end[2],
                        begin=flow.begin,
                        end=flow.end,
                        **rate_option,
                    )
            # write trip into xml format
            if traffic.trips:
                self.write_trip_xml(traffic, doc, fill_in_route_gaps)

        with open(route_path, "w") as f:
            f.write(
                indent(
                    doc.getvalue(), indentation="    ", newline="\r\n", indent_text=True
                )
            )

    def write_trip_xml(self, traffic, doc, fill_in_gaps):
        """Writes a trip spec into a route file. Typically this would be the source
        data to SUMO's DUAROUTER.
        """
        # Make sure all routes are "resolved" (e.g. `RandomRoute` are converted to
        # `Route`) so that we can write them all to file.
        resolved_routes = {}
        for route in {trip.route for trip in traffic.trips}:
            resolved_routes[route] = self.resolve_route(route, fill_in_gaps)

        for route in set(resolved_routes.values()):
            doc.stag("route", id=route.id + "trip", edges=" ".join(route.roads))

            # We don't de-dup flows since defining the same flow multiple times should
            # create multiple traffic flows. Since IDs can't be reused, we also unique
            # them here.
        for trip_idx, trip in enumerate(traffic.trips):
            route = resolved_routes[trip.route]
            actor = trip.actor
            doc.stag(
                "vehicle",
                id="{}".format(trip.vehicle_name),
                type=actor.id,
                route=route.id + "trip",
                depart=trip.depart,
                departLane=route.begin[1],
                departPos=route.begin[2],
                departSpeed=actor.depart_speed,
                arrivalLane=route.end[1],
                arrivalPos=route.end[2],
            )

    def _cache_road_network(self):
        if not self._road_network:
            from smarts.core.sumo_road_network import SumoRoadNetwork

            map_spec = types.MapSpec(self._road_network_path)
            self._road_network = SumoRoadNetwork.from_spec(map_spec)

    def resolve_edge_length(self, edge_id, lane_idx):
        """Determine the length of the given lane on an edge.
        Args:
            edge_id: The edge id of the road segment.
            lane_idx: The index of a lane.
        Returns:
            The length of the lane (same as the length of the edge.)
        """
        self._cache_road_network()
        lane = self._road_network.road_by_id(edge_id).lanes[lane_idx]
        return lane.length

    def _map_for_route(self, route) -> RoadMap:
        map_spec = route.map_spec
        if not map_spec:
            # XXX: Spacing is crudely "large enough" so we're less likely overlap vehicles
            lp_spacing = 2.0
            if self._scenario_map_spec:
                map_spec = replace(
                    self._scenario_map_spec, lanepoint_spacing=lp_spacing
                )
            else:
                map_spec = types.MapSpec(
                    self._road_network_path, lanepoint_spacing=lp_spacing
                )
        road_map, _ = map_spec.builder_fn(map_spec)
        return road_map

    def _fill_in_gaps(self, route: types.Route) -> types.Route:
        # TODO:  do this at runtime so each vehicle on the flow can take a different variation of the route ?
        # TODO:  or do it like SUMO and generate a huge *.rou.xml file instead ?
        road_map = self._map_for_route(route)
        start_road = road_map.road_by_id(route.begin[0])
        end_road = road_map.road_by_id(route.end[0])
        vias = [road_map.road_by_id(via) for via in route.via]
        routes = road_map.generate_routes(start_road, end_road, vias)
        if not routes:
            raise InvalidRoute(
                "Could not find route that starts at {route.begin[0]}, ends at {route.end[0]} and includes {route.vias}."
            )
        return replace(
            route, via=tuple((road.road_id for road in routes[0].roads[1:-1]))
        )

    def resolve_route(self, route, fill_in_gaps: bool) -> types.Route:
        """Attempts to fill in the route between the begining and end specified in the initial
         route.

        :param route: An incomplete route.
        :return: A complete route listing all road segments it passes through.
        """
        if not isinstance(route, types.RandomRoute):
            return self._fill_in_gaps(route) if fill_in_gaps else route

        if not self._random_route_generator:
            road_map = self._map_for_route(route)
            # Lazy-load to improve performance when not using random route generation.
            self._random_route_generator = RandomRouteGenerator(road_map)

        return next(self._random_route_generator)

    @property
    def road_network(self):
        """Retrieves the road network this generator is associated with."""
        self._cache_road_network()
        return self._road_network

    def _resolve_log_dir(self, log_dir):
        if log_dir is None:
            log_dir = make_dir_in_smarts_log_dir("_duarouter_routing")

        return os.path.abspath(log_dir)
