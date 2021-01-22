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
import atexit
import logging
import numpy as np
import os
import random
import subprocess
import time

from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon, box as shapely_box
from typing import List, Sequence

from smarts.core import gen_id
from smarts.core.colors import SceneColors
from smarts.core.coordinates import Heading, Pose
from smarts.core.provider import ProviderState, ProviderTLS, ProviderTrafficLight
from smarts.core.vehicle import VEHICLE_CONFIGS, VehicleState
from smarts.core.utils import networking
from smarts.core.utils.logging import surpress_stdout
from smarts.core.utils.sumo import SUMO_PATH, traci

import traci.constants as tc
from traci.exceptions import FatalTraCIError, TraCIException


class SumoTrafficSimulation:
    """
    Args:
        net_file:
            path to sumo .net.xml file
        headless:
            False to run with `sumo-gui`. True to run with `sumo`
        time_resolution:
            SUMO simulation is descretized into steps of `time_resolution` seconds
            WARNING:
                Since our interface(TRACI) to SUMO is delayed by one simulation step,
                setting a higher time resolution may lead to unexpected artifacts
    """

    def __init__(
        self,
        headless=True,
        time_resolution=0.1,
        num_external_sumo_clients=0,
        sumo_port=None,
        auto_start=True,
        endless_traffic=True,
        allow_reload=True,
        debug=True,
    ):
        self._log = logging.getLogger(self.__class__.__name__)

        self._debug = debug
        self._scenario = None
        self._log_file = None
        self._time_resolution = time_resolution
        self._headless = headless
        self._cumulative_sim_seconds = 0
        self._non_sumo_vehicle_ids = set()
        self._sumo_vehicle_ids = set()
        self._is_setup = False
        self._last_trigger_time = -1000000
        self._num_dynamic_ids_used = 0
        self._traci_conn = None
        self._sumo_proc = None
        self._num_clients = 1 + num_external_sumo_clients
        self._sumo_port = sumo_port
        self._auto_start = auto_start
        self._endless_traffic = endless_traffic
        self._to_be_teleported = dict()
        self._reserved_areas = dict()
        self._allow_reload = allow_reload

    def __repr__(self):
        return f"""SumoTrafficSim(
  _scenario={repr(self._scenario)},
  _time_resolution={self._time_resolution},
  _headless={self._headless},
  _cumulative_sim_seconds={self._cumulative_sim_seconds},
  _non_sumo_vehicle_ids={self._non_sumo_vehicle_ids},
  _sumo_vehicle_ids={self._sumo_vehicle_ids},
  _is_setup={self._is_setup},
  _last_trigger_time={self._last_trigger_time},
  _num_dynamic_ids_used={self._num_dynamic_ids_used},
  _traci_conn={repr(self._traci_conn)}
)"""

    def __str__(self):
        return repr(self)

    def destroy(self):
        self._close_traci_and_pipes()
        if not self._is_setup:
            return
        self._sumo_proc.terminate()
        self._sumo_proc.wait()

    def _initialize_traci_conn(self, num_retries=5):
        # TODO: inline sumo or process pool
        # the retries are to deal with port collisions
        #   since the way we start sumo here has a race condition on
        #   each spawned process claiming a port
        for _ in range(num_retries):
            self._close_traci_and_pipes()

            sumo_port = self._sumo_port
            if sumo_port is None:
                sumo_port = networking.find_free_port()

            sumo_binary = "sumo" if self._headless else "sumo-gui"
            sumo_cmd = [
                os.path.join(SUMO_PATH, "bin", sumo_binary),
                "--remote-port=%s" % sumo_port,
                *self._base_sumo_load_params(),
            ]

            self._log.debug("Starting sumo process:\n\t %s", sumo_cmd)
            self._sumo_proc = subprocess.Popen(
                sumo_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            time.sleep(0.05)  # give SUMO time to start
            try:
                with surpress_stdout():
                    self._traci_conn = traci.connect(
                        sumo_port,
                        numRetries=100,
                        proc=self._sumo_proc,
                        waitBetweenRetries=0.05,
                    )  # SUMO must be ready within 5 seconds

                try:
                    assert (
                        self._traci_conn.getVersion()[0] >= 20
                    ), "TraCI API version must be >= 20 (SUMO 1.5.0)"
                # We will retry since this is our first sumo command
                except FatalTraCIError:
                    logging.debug("Connection closed. Retrying...")
                    self._close_traci_and_pipes()
                    continue
            except ConnectionRefusedError:
                logging.debug(
                    "Connection refused. Tried to connect to unpaired TraCI client."
                )
                self._close_traci_and_pipes()
                continue

            # It is mandatory to set order when using multiple clients.
            self._traci_conn.setOrder(0)

            break

        try:
            self._traci_conn.getVersion()
        except Exception as e:
            logging.error(
                f"""Failed to initialize SUMO
                Your scenario might not be configured correctly or
                you were trying to initialize many SUMO instances at
                once and we were not able to assign unique port
                numbers to all SUMO processes.

                Check {self._log_file} for hints"""
            )
            raise e

        self._log.debug("Finished starting sumo process")

    def _base_sumo_load_params(self):
        load_params = [
            "--num-clients=%d" % self._num_clients,
            "--net-file=%s" % self._scenario.net_filepath,
            "--quit-on-end",
            "--log=%s" % self._log_file,
            "--error-log=%s" % self._log_file,
            "--no-step-log",
            "--no-warnings=1",
            "--seed=%s" % random.randint(0, 2147483648),
            "--time-to-teleport=%s" % -1,
            "--collision.check-junctions=true",
            "--collision.action=none",
            "--lanechange.duration=3.0",
            # TODO: 1--lanechange.duration1 or 1--lateral-resolution`, in combination with `route_id`,
            # causes lane change crashes as of SUMO 1.6.0.
            # Controlling vehicles that have been added to the simulation with a route causes
            # lane change related crashes.
            # "--lateral-resolution=100",  # smooth lane changes
            "--step-length=%f" % self._time_resolution,
            "--default.action-step-length=%f" % self._time_resolution,
            "--begin=0",  # start simulation at time=0
            "--end=31536000",  # keep the simulation running for a year
        ]

        if self._auto_start:
            load_params.append("--start")

        if self._scenario.route_files_enabled:
            load_params.append("--route-files={}".format(self._scenario.route_filepath))

        return load_params

    def setup(self, scenario) -> ProviderState:
        self._log.debug("Setting up SumoTrafficSim %s" % self)
        assert not self._is_setup, (
            "Can't setup twice, %s, see teardown()" % self._is_setup
        )

        # restart sumo process only when map file changes
        if self._scenario and self._scenario.net_file_hash == scenario.net_file_hash:
            restart_sumo = False
        else:
            restart_sumo = True

        self._scenario = scenario
        self._log_file = scenario.unique_sumo_log_file()

        if restart_sumo:
            self._initialize_traci_conn()
        elif self._allow_reload:
            self._traci_conn.load(self._base_sumo_load_params())

        assert self._traci_conn is not None, "No active traci conn"

        self._traci_conn.simulation.subscribe(
            [tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS]
        )

        for tls_id in self._traci_conn.trafficlight.getIDList():
            self._traci_conn.trafficlight.subscribe(
                tls_id, [tc.TL_RED_YELLOW_GREEN_STATE, tc.TL_CONTROLLED_LINKS]
            )

        # XXX: SUMO caches the previous subscription results. Calling `simulationStep`
        #      effectively flushes the results. We need to use epsilon instead of zero
        #      as zero will step according to a default (non-zero) step-size.
        self.step({}, 1e-6, 0)

        self._is_setup = True

        return self._compute_provider_state()

    def _close_traci_and_pipes(self):
        if self._sumo_proc:
            self._sumo_proc.stdin.close()
            self._sumo_proc.stdout.close()
            self._sumo_proc.stderr.close()

        if self._traci_conn:
            self._traci_conn.close()
            self._traci_conn = None

    def _remove_all_vehicles(self):
        for vehicle_id in self._non_sumo_vehicle_ids.union(self._sumo_vehicle_ids):
            self._traci_conn.vehicle.remove(vehicle_id)

    def teardown(self):
        self._log.debug("Tearing down SUMO traffic sim %s" % self)
        if not self._is_setup:
            self._log.debug("Nothing to teardown")
            return

        assert self._is_setup

        self._remove_all_vehicles()
        if self._allow_reload:
            self._cumulative_sim_seconds = 0
        self._non_sumo_vehicle_ids = set()
        self._sumo_vehicle_ids = set()
        self._is_setup = False
        self._num_dynamic_ids_used = 0
        self._to_be_teleported = dict()
        self._reserved_areas = dict()

    @property
    def action_spaces(self):
        # Unify interfaces with other providers
        return {}

    def reset(self):
        # Unify interfaces with other providers
        pass

    def step(self, provider_actions, dt, elapsed_sim_time) -> ProviderState:
        """
        Args:
            dt: time (in seconds) to simulate during this simulation step
            managed_vehicles: dict of {vehicle_id: (x, y, heading)}
                !! The vehicle state should represent the state of the
                !! vehicles at the start of the current simulation step
        Returns:
            ProviderState representing the state of the SUMO simulation
        """
        # we tell SUMO to step through dt more seconds of the simulation
        self._cumulative_sim_seconds += dt
        self._traci_conn.simulationStep(self._cumulative_sim_seconds)

        return self._compute_provider_state()

    def sync(self, provider_state: ProviderState):
        provider_vehicles = {v.vehicle_id: v for v in provider_state.vehicles}
        external_vehicles = [v for v in provider_state.vehicles if v.source != "SUMO"]
        external_vehicle_ids = {v.vehicle_id for v in external_vehicles}

        # Represents current state
        traffic_vehicle_states = self._traci_conn.vehicle.getAllSubscriptionResults()
        traffic_vehicle_ids = set(traffic_vehicle_states)

        # State / ownership changes
        external_vehicles_that_have_left = (
            self._non_sumo_vehicle_ids - external_vehicle_ids - traffic_vehicle_ids
        )
        external_vehicles_that_have_joined = (
            external_vehicle_ids - self._non_sumo_vehicle_ids - traffic_vehicle_ids
        )
        vehicles_that_have_become_external = (
            traffic_vehicle_ids & external_vehicle_ids - self._non_sumo_vehicle_ids
        )
        # XXX: They may have become internal because they've been relinquished or
        #      because they've been destroyed from a collision. Presently we're not
        #      differentiating and will take over as social vehicles regardless.
        vehicles_that_have_become_internal = (
            self._non_sumo_vehicle_ids - external_vehicle_ids
        ) & traffic_vehicle_ids

        log = ""
        if external_vehicles_that_have_left:
            log += (
                f"external_vehicles_that_have_left={external_vehicles_that_have_left}\n"
            )
        if external_vehicles_that_have_joined:
            log += f"external_vehicles_that_have_joined={external_vehicles_that_have_joined}\n"
        if vehicles_that_have_become_external:
            log += f"vehicles_that_have_become_external={vehicles_that_have_become_external}\n"
        if vehicles_that_have_become_internal:
            log += f"vehicles_that_have_become_internal={vehicles_that_have_become_internal}\n"

        if log:
            self._log.debug(log)

        for vehicle_id in external_vehicles_that_have_left:
            self._log.debug("Non SUMO vehicle %s left simulation", vehicle_id)
            self._non_sumo_vehicle_ids.remove(vehicle_id)
            self._traci_conn.vehicle.remove(vehicle_id)

        for vehicle_id in external_vehicles_that_have_joined:
            dimensions = provider_vehicles[vehicle_id].dimensions
            self._create_vehicle(vehicle_id, dimensions)

        # update the state of all current managed vehicles
        for vehicle_id in self._non_sumo_vehicle_ids:
            provider_vehicle = provider_vehicles[vehicle_id]

            pos, sumo_heading = provider_vehicle.pose.as_sumo(
                provider_vehicle.dimensions.length, Heading(0)
            )

            # See https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#move_to_xy_0xb4
            # for flag values
            try:
                self._move_vehicle(
                    provider_vehicle.vehicle_id,
                    pos,
                    sumo_heading,
                    provider_vehicle.speed,
                )
            except TraCIException as e:
                # Likely as a result of https://github.com/eclipse/sumo/issues/3993
                # the vehicle got removed because we skipped a moveToXY call between
                # internal stepSimulations, so we add the vehicle back here.
                self._log.warning(
                    "Attempted to (TraCI) SUMO.moveToXY(...) on missing "
                    f"vehicle(id={vehicle_id})"
                )
                self._create_vehicle(vehicle_id, provider_vehicle.dimensions)
                self._move_vehicle(
                    provider_vehicle.vehicle_id,
                    pos,
                    sumo_heading,
                    provider_vehicle.speed,
                )

        for vehicle_id in vehicles_that_have_become_external:
            self._traci_conn.vehicle.setColor(
                vehicle_id, SumoTrafficSimulation._social_agent_vehicle_color()
            )
            self._non_sumo_vehicle_ids.add(vehicle_id)

        for vehicle_id in vehicles_that_have_become_internal:
            self._traci_conn.vehicle.setColor(
                vehicle_id, SumoTrafficSimulation._social_vehicle_color()
            )
            self._non_sumo_vehicle_ids.remove(vehicle_id)
            # Let sumo take over speed again
            self._traci_conn.vehicle.setSpeed(vehicle_id, -1)

        if self._endless_traffic:
            self._reroute_vehicles(traffic_vehicle_states)
            self._teleport_exited_vehicles()

    @staticmethod
    def _ego_agent_vehicle_color():
        return np.array(SceneColors.Agent.value[:3]) * 255

    @staticmethod
    def _social_agent_vehicle_color():
        return np.array(SceneColors.SocialAgent.value[:3]) * 255

    @staticmethod
    def _social_vehicle_color():
        return np.array(SceneColors.SocialVehicle.value[:3]) * 255

    def _move_vehicle(self, vehicle_id, position, heading, speed):
        x, y, _ = position
        self._traci_conn.vehicle.moveToXY(
            vehID=vehicle_id,
            edgeID="",  # let sumo choose the edge
            lane=-1,  # let sumo choose the lane
            x=x,
            y=y,
            angle=heading,  # only used for visualizing in sumo-gui
            keepRoute=0b010,
        )
        self._traci_conn.vehicle.setSpeed(vehicle_id, speed)

    def _create_vehicle(self, vehicle_id, dimensions):
        assert (
            type(vehicle_id) == str
        ), f"SUMO expects string ids: {vehicle_id} is a {type(vehicle_id)}"

        self._log.debug("Non SUMO vehicle %s joined simulation", vehicle_id)
        self._non_sumo_vehicle_ids.add(vehicle_id)
        self._traci_conn.vehicle.add(
            vehID=vehicle_id,
            routeID="",  # we don't care which route this vehicle is on
        )

        # TODO: Vehicle Id should not be using prefixes this way
        if vehicle_id.startswith("social-agent"):
            # This is based on ID convention
            vehicle_color = SumoTrafficSimulation._social_agent_vehicle_color()
        else:
            vehicle_color = SumoTrafficSimulation._ego_agent_vehicle_color()

        self._traci_conn.vehicle.setColor(vehicle_id, vehicle_color)

        # Directly below are two of the main factors that affect vehicle secure gap for
        # purposes of determining the safety gaps that SUMO vehicles will abide by. The
        # remaining large factor is vehicle speed.
        # See:
        # http://sumo-user-mailing-list.90755.n8.nabble.com/sumo-user-Questions-on-SUMO-Built-In-Functions-getSecureGap-amp-brakeGap-td3254.html
        # Set the controlled vehicle's time headway in seconds
        self._traci_conn.vehicle.setTau(vehicle_id, 4)
        # Set the controlled vehicle's maximum natural deceleration in m/s
        self._traci_conn.vehicle.setDecel(vehicle_id, 6)

        # setup the vehicle size
        self._traci_conn.vehicle.setLength(vehicle_id, dimensions.length)
        self._traci_conn.vehicle.setWidth(vehicle_id, dimensions.width)
        self._traci_conn.vehicle.setHeight(vehicle_id, dimensions.height)

    def _compute_provider_state(self) -> ProviderState:
        return ProviderState(
            vehicles=self._compute_traffic_vehicles(),
            traffic_light_systems=self._compute_traffic_lights(),
        )

    def _compute_traffic_vehicles(self) -> List[VehicleState]:
        sub_results = self._traci_conn.simulation.getSubscriptionResults()

        if sub_results is None or sub_results == {}:
            return {}

        # New social vehicles that have entered the map
        newly_departed_sumo_traffic = [
            vehicle_id
            for vehicle_id in sub_results[tc.VAR_DEPARTED_VEHICLES_IDS]
            if vehicle_id not in self._non_sumo_vehicle_ids
        ]

        reserved_areas = [position for position in self._reserved_areas.values()]

        # Subscribe to all vehicles to reduce repeated traci calls
        for vehicle_id in newly_departed_sumo_traffic:
            self._traci_conn.vehicle.subscribe(
                vehicle_id,
                [
                    tc.VAR_POSITION,  # Decimal=66,  Hex=0x42
                    tc.VAR_ANGLE,  # Decimal=67,  Hex=0x43
                    tc.VAR_SPEED,  # Decimal=64,  Hex=0x40
                    tc.VAR_VEHICLECLASS,  # Decimal=73,  Hex=0x49
                    tc.VAR_ROUTE_INDEX,  # Decimal=105, Hex=0x69
                    tc.VAR_EDGES,  # Decimal=84,  Hex=0x54
                    tc.VAR_TYPE,  # Decimal=79,  Hex=0x4F
                    tc.VAR_LENGTH,  # Decimal=68,  Hex=0x44
                    tc.VAR_WIDTH,  # Decimal=77,  Hex=0x4d
                ],
            )

        sumo_vehicle_state = self._traci_conn.vehicle.getAllSubscriptionResults()

        for vehicle_id in newly_departed_sumo_traffic:
            other_vehicle_shape = self._shape_of_vehicle(sumo_vehicle_state, vehicle_id)

            violates_reserved_area = False
            for reserved_area in reserved_areas:
                if reserved_area.intersects(other_vehicle_shape):
                    violates_reserved_area = True
                    break

            if violates_reserved_area:
                self._traci_conn.vehicle.remove(vehicle_id)
                sumo_vehicle_state.pop(vehicle_id)
                continue

            self._log.debug("SUMO vehicle %s entered simulation", vehicle_id)

        # Non-sumo vehicles will show up the step after the sync where the non-sumo vehicle is
        # added.
        newly_departed_non_sumo_vehicles = [
            vehicle_id
            for vehicle_id in sub_results[tc.VAR_DEPARTED_VEHICLES_IDS]
            if vehicle_id not in newly_departed_sumo_traffic
        ]

        for vehicle_id in newly_departed_non_sumo_vehicles:
            if vehicle_id in self._reserved_areas:
                del self._reserved_areas[vehicle_id]

        self._sumo_vehicle_ids = (
            set(sumo_vehicle_state.keys()) - self._non_sumo_vehicle_ids
        )
        provider_vehicles = []

        # batched conversion of positions to numpy arrays
        front_bumper_positions = np.array(
            [
                sumo_vehicle[tc.VAR_POSITION]
                for sumo_vehicle in sumo_vehicle_state.values()
            ]
        ).reshape(-1, 2)

        for i, (sumo_id, sumo_vehicle) in enumerate(sumo_vehicle_state.items()):
            # XXX: We can safely rely on iteration order over dictionaries being
            #      stable on py3.7.
            #      See: https://www.python.org/downloads/release/python-370/
            #      "The insertion-order preservation nature of dict objects is now an
            #      official part of the Python language spec."
            front_bumper_pos = front_bumper_positions[i]
            heading = Heading.from_sumo(sumo_vehicle[tc.VAR_ANGLE])
            speed = sumo_vehicle[tc.VAR_SPEED]
            vehicle_type = sumo_vehicle[tc.VAR_VEHICLECLASS]
            dimensions = VEHICLE_CONFIGS[vehicle_type].dimensions
            provider_vehicles.append(
                VehicleState(
                    # XXX: In the case of the SUMO traffic provider, the vehicle ID is
                    #      the sumo ID is the actor ID.
                    vehicle_id=sumo_id,
                    vehicle_type=vehicle_type,
                    pose=Pose.from_front_bumper(
                        front_bumper_pos, heading, dimensions.length
                    ),
                    dimensions=dimensions,
                    speed=speed,
                    source="SUMO",
                )
            )

        return provider_vehicles

    def _teleport_exited_vehicles(self):
        sub_results = self._traci_conn.simulation.getSubscriptionResults()
        if not sub_results:
            return

        exited_sumo_traffic = [
            vehicle_id
            for vehicle_id in sub_results[tc.VAR_ARRIVED_VEHICLES_IDS]
            if vehicle_id not in self._non_sumo_vehicle_ids
        ]
        for v_id in exited_sumo_traffic:
            if v_id in self._to_be_teleported:
                route = self._to_be_teleported[v_id]["route"]
                type_id = self._to_be_teleported[v_id]["type_id"]
                self._teleport_vehicle(v_id, route, 0, type_id)

    def _teleport_vehicle(self, vehicle_id, route, lane_offset, type_id):
        self._log.debug(
            f"Teleporting {vehicle_id} to lane_offset={lane_offset} route={route}"
        )
        spawn_edge = self._scenario.road_network.graph.getEdge(route[0])
        lane_index = random.randint(0, len(spawn_edge.getLanes()) - 1)
        self._emit_vehicle_by_route(vehicle_id, route, lane_index, lane_offset, type_id)

    def _reroute_vehicles(self, vehicle_states):
        for vehicle_id, state in vehicle_states.items():
            if vehicle_id not in self._sumo_vehicle_ids:
                continue

            route_index = state[tc.VAR_ROUTE_INDEX]
            route_edges = state[tc.VAR_EDGES]
            type_id = state[tc.VAR_TYPE]

            if route_index != len(route_edges) - 1:
                # The vehicle is not in the last route edge.
                continue

            # Check if these edges forms a loop.
            from_edge = self._scenario.road_network.graph.getEdge(route_edges[-1])
            to_edge = self._scenario.road_network.graph.getEdge(route_edges[0])
            next_edges = from_edge.getOutgoing().keys()
            if to_edge not in next_edges:
                # Reroute only if it's loop, otherwise, teleport the vehicle.
                self._to_be_teleported[vehicle_id] = {
                    "route": route_edges,
                    "type_id": type_id,
                }
                continue
            # The first edge in the list has to be the one that the vehicle
            # is in at the moment, which is the last edge in current route_edges.
            new_route_edges = route_edges[-1:] + route_edges
            self._traci_conn.vehicle.setRoute(vehicle_id, new_route_edges)

    def _compute_traffic_lights(self) -> List[ProviderTLS]:
        """TraCI will automatically generate TLS programs if none was specified
        according to the net/program. To support this we opt to use TraCI instead
        of the sumolib interface for TLS support.
        """

        sub_results = self._traci_conn.trafficlight.getSubscriptionResults(None)

        tlss = []
        if not sub_results:
            return tlss

        for tls_id in sub_results:
            light_states = sub_results[tls_id][tc.TL_RED_YELLOW_GREEN_STATE]
            links = sub_results[tls_id][tc.TL_CONTROLLED_LINKS]

            traffic_lights = []
            for link, state in zip(links, light_states):
                if not link:
                    continue
                lane_start, lane_end, lane_via = [
                    self._scenario.road_network.lane_by_id(lane) for lane in link[0]
                ]
                traffic_lights.append(
                    ProviderTrafficLight(
                        lane_in=lane_start,
                        lane_via=lane_via,
                        lane_out=lane_end,
                        state=state,
                    )
                )
            tlss.append(ProviderTLS(tls_id, traffic_lights))

        return tlss

    def _unique_id(self):
        route_id = "hiway_id_%s" % self._num_dynamic_ids_used
        self._num_dynamic_ids_used += 1
        return route_id

    def vehicle_route(self, vehicle_id) -> Sequence[str]:
        return self._traci_conn.vehicle.getRoute(vehicle_id)

    def reserve_traffic_location_for_vehicle(
        self, vehicle_id: str, reserved_location: Polygon,
    ):
        """Reserve an area around a location where vehicles cannot spawn until a given vehicle
        is added.

        Args:
            vehicle_id: The vehicle to wait for.
            reserved_location: The space the vehicle takes up.
        """
        self._reserved_areas[vehicle_id] = reserved_location

    def remove_traffic_vehicle(self, vehicle_id: str):
        self._traci_conn.vehicle.remove(vehicle_id)
        self._sumo_vehicle_ids.remove(vehicle_id)

    def _shape_of_vehicle(self, sumo_vehicle_state, vehicle_id):
        p = sumo_vehicle_state[vehicle_id][tc.VAR_POSITION]
        length = sumo_vehicle_state[vehicle_id][tc.VAR_LENGTH]
        width = sumo_vehicle_state[vehicle_id][tc.VAR_WIDTH]
        heading = Heading.from_sumo(sumo_vehicle_state[vehicle_id][tc.VAR_ANGLE])

        poly = shapely_box(p[0] - width * 0.5, p[1] - length, p[0] + width * 0.5, p[1],)
        return shapely_rotate(poly, heading, use_radians=True)

    def _emit_vehicle_by_route(
        self, vehicle_id, route, lane_index, lane_offset, type_id="DEFAULT_VEHTYPE"
    ):
        route_id = f"route-{gen_id()}"
        self._traci_conn.route.add(route_id, route)
        self._traci_conn.vehicle.add(
            vehicle_id,
            route_id,
            typeID=type_id,
            departPos=lane_offset,
            departLane=lane_index,
        )
        return vehicle_id

    def _emit_vehicle_near_position(self, position, vehicle_id=None) -> str:
        wp = self._scenario.waypoints.closest_waypoint(position)
        lane = self._scenario.road_network.lane_by_id(wp.lane_id)
        offset_in_lane = self._scenario.road_network.offset_into_lane(
            lane, tuple(wp.pos)
        )

        if not vehicle_id:
            vehicle_id = self._unique_id()

        # XXX: Do not give this a route or it will crash on `moveTo` calls
        self._traci_conn.vehicle.add(
            vehicle_id, "", departPos=offset_in_lane, departLane=wp.lane_index,
        )

        self._traci_conn.vehicle.moveToXY(
            vehID=vehicle_id,
            edgeID="",  # let sumo choose the edge
            lane=-1,  # let sumo choose the lane
            x=position[0],
            y=position[1],
            # angle=sumo_heading,  # only used for visualizing in sumo-gui
            keepRoute=0b000,  # On lane
        )
        return vehicle_id
