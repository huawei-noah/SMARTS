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
import math
from typing import List, Sequence, Tuple

import numpy as np

from .agent_interface import AgentBehavior
from .coordinates import Heading, Pose
from .planner import Planner, Waypoint
from .road_map import RoadMap
from .scenario import Mission
from .sumo_lanepoints import LanePoint, LinkedLanePoint
from .sumo_road_network import SumoRoadNetwork
from .utils.math import (
    evaluate_bezier as bezier,
    fast_quaternion_from_angle,
    inplace_unwrap,
    radians_to_vec,
    vec_2d,
    vec_to_radians,
)
from .vehicle import Vehicle


class SumoPlanner(Planner):
    def __init__(self, road_map: RoadMap, agent_behavior: AgentBehavior = None):
        super().__init__(road_map, agent_behavior)
        assert isinstance(self._road_map, SumoRoadNetwork)
        self._log = logging.getLogger(self.__class__.__name__)
        self._lanepoints = self._road_map.lanepoints
        self._waypoints_cache = SumoPlanner._WaypointsCache()
        self._task_is_triggered = False
        # TODO: These variables should be put in an appropriate place.
        self._uturn_initial_heading = 0
        self._uturn_initial_distant = 0
        self._uturn_initial_velocity = 0
        self._uturn_initial_height = 0
        self._insufficient_initial_distant = False
        self._uturn_initial_position = 0
        self._uturn_is_initialized = False
        self._prev_kyber_x_position = None
        self._prev_kyber_y_position = None
        self._first_uturn = True

    def waypoint_paths(
        self,
        vehicle: Vehicle,
        lookahead: int,
        within_radius: float = 5,
        constrain_to_route: bool = True,
        context=None,
    ) -> List[List[Waypoint]]:
        if self._mission.task is not None:
            if isinstance(self._mission.task, UTurn):
                return self._uturn_waypoints(vehicle, context)
            elif isinstance(self._mission.task, CutIn):
                return self._cut_in_waypoints(vehicle, context)

        pose = vehicle.pose
        if constrain_to_route:
            assert (
                self._did_plan
            ), "Must call plan(...) before being able to use a route."
            road_ids = self._road_ids(pose)
            if road_ids:
                return self._waypoint_paths_along_route(
                    pose.position, lookahead, road_ids
                )

        return self._waypoint_paths_at(pose, lookahead, within_radius)

    def waypoint_paths_on_lane_at(
        self, pose: Pose, lane_id: str, lookahead: int, constrain_to_route: bool = True
    ) -> List[List[Waypoint]]:
        road_ids = None
        if constrain_to_route:
            assert (
                self._did_plan
            ), "Must call plan(...) before being able to use a route."
            road_ids = self._road_ids(pose, lane_id)

        return self._waypoint_paths_on_lane_at(
            pose.position, lane_id, lookahead, road_ids
        )

    def _road_ids(self, pose: Pose, lane_id: str = None):
        if self._mission.has_fixed_route:
            return [road.road_id for road in self._route.roads]

        # Filter lanepoints to the internal lane the vehicle is driving on to deal w/
        # non-fixed routes (e.g. endless missions). This is so that the lanepoints don't
        # jump between junction connections.
        if lane_id is None:
            # We take the 10 closest lanepoints to then filter down to that which has
            # the closest heading. This way we get the lanepoint on our lane instead of
            # a potentially closer lane that is on a different junction connection.
            closest_lps = self._lanepoints.closest_lanepoints(
                poses=[pose], within_radius=None, maximum_count=10
            )
            closest_lps = sorted(
                closest_lps, key=lambda lp: abs(pose.heading - lp.pose.heading)
            )
            lane_id = closest_lps[0].lane.lane_id

        lane = self._road_map.lane_by_id(lane_id)
        if not lane.in_junction:
            return []

        road = lane.road
        road_ids = [road.road_id]

        next_roads = list(road.outgoing_roads.keys())
        assert (
            len(next_roads) <= 1
        ), "A junction is expected to have <= 1 outgoing roads"
        if next_roads:
            road_ids.append(next_roads[0].road_id)

        return road_ids

    def _cut_in_waypoints(self, vehicle: Vehicle, sim) -> List[List[Waypoint]]:
        pose = vehicle.pose
        aggressiveness = self._agent_behavior.aggressiveness or 0

        neighborhood_vehicles = sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=850
        )

        position = Point(*pose.position)
        lane = self._road_map.nearest_lane(position)

        if not neighborhood_vehicles or sim.elapsed_sim_time < 1:
            return []

        target_vehicle = neighborhood_vehicles[0]
        target_position = Point(*target_vehicle.pose.position)

        if (self._prev_kyber_x_position is None) and (
            self._prev_kyber_y_position is None
        ):
            self._prev_kyber_x_position = target_position.x
            self._prev_kyber_y_position = target_position.y

        velocity_vector = np.array(
            [
                (-self._prev_kyber_x_position + target_position.x) / sim.timestep_sec,
                (-self._prev_kyber_y_position + target_position.y) / sim.timestep_sec,
            ]
        )
        target_velocity = np.dot(
            velocity_vector, radians_to_vec(target_vehicle.pose.heading)
        )

        self._prev_kyber_x_position = target_position.x
        self._prev_kyber_y_position = target_position.y

        target_lane = self._road_map.nearest_lane(target_position)

        offset = lane.offset_along_lane(position)
        target_offset = target_lane.offset_along_lane(target_position)

        # cut-in offset should consider the aggressiveness and the speed
        # of the other vehicle.

        cut_in_offset = np.clip(20 - aggressiveness, 10, 20)
        p0 = pose.position[:2]

        if (
            abs(offset - (cut_in_offset + target_offset)) > 1
            and lane.lane_id != target_lane.lane_id
            and self._task_is_triggered is False
        ):
            nei_wps = self._waypoint_paths_on_lane_at(p0, lane.lane_id, 60)
            speed_limit = np.clip(
                np.clip(
                    (target_velocity * 1.1)
                    - 6 * (offset - (cut_in_offset + target_offset)),
                    0.5 * target_velocity,
                    2 * target_velocity,
                ),
                0.5,
                30,
            )
        else:
            self._task_is_triggered = True
            nei_wps = self._waypoint_paths_on_lane_at(p0, target_lane.lane_id, 60)

            cut_in_speed = target_velocity * 2.3

            speed_limit = cut_in_speed

            # 1.5 m/s is the threshold for speed offset. If the vehicle speed
            # is less than target_velocity plus this offset then it will not
            # perform the cut-in task and instead the speed of the vehicle is
            # increased.
            if vehicle.speed < target_velocity + 1.5:
                nei_wps = self._waypoint_paths_on_lane_at(p0, lane.lane_id, 60)
                speed_limit = np.clip(target_velocity * 2.1, 0.5, 30)
                self._task_is_triggered = False

        p_temp = nei_wps[0][len(nei_wps[0]) // 3].pos
        p1 = p_temp
        p2 = nei_wps[0][2 * len(nei_wps[0]) // 3].pos

        p3 = nei_wps[0][-1].pos
        p_x, p_y = bezier([p0, p1, p2, p3], 20)
        trajectory = []
        prev = p0
        for i in range(len(p_x)):
            pos = np.array([p_x[i], p_y[i]])
            heading = Heading(vec_to_radians(pos - prev))
            prev = pos
            lane = self._road_map.nearest_lane(pos)
            if lane is None:
                continue
            lane_id = lane.lane_id
            lane_index = lane_id.split("_")[-1]
            width = lane.width

            wp = Waypoint(
                pose=Pose(
                    position=pos, orientation=fast_quaternion_from_angle(heading)
                ),
                lane_width=width,
                speed_limit=speed_limit,
                lane_id=lane_id,
                lane_index=lane_index,
            )
            trajectory.append(wp)
        return [trajectory]

    def _uturn_waypoints(self, vehicle: Vehicle, sim) -> List[List[Waypoint]]:
        # TODO: 1. Need to revisit the approach to calculate the U-Turn trajectory.
        #       2. Wrap this method in a helper.

        ## the position of ego car is here: [x, y]
        pose = vehicle.pose
        ego_position = Point(*pose.position)
        ego_lane = self._road_map.nearest_lane(ego_position)
        ego_wps = self._waypoint_paths_on_lane_at(ego_position, ego_lane.lane_id, 60)
        if self._mission.task.initial_speed is None:
            default_speed = ego_wps[0][0].speed_limit
        else:
            default_speed = self._mission.task.initial_speed
        ego_wps_des_speed = []
        for px in range(len(ego_wps[0])):
            new_wp = replace(ego_wps[0][px], speed_limit=default_speed)
            ego_wps_des_speed.append(new_wp)

        ego_wps_des_speed = [ego_wps_des_speed]
        neighborhood_vehicles = sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=140
        )

        if not neighborhood_vehicles:
            return ego_wps_des_speed

        n_lane = self._road_map.nearest_lane(
            Point(*neighborhood_vehicles[0].pose.position)
        )
        start_lane = self._road_map.nearest_lane(
            Point(*self._mission.start.position),
            include_junctions=False,
        )
        oncoming_lanes = start_lane.oncoming_lanes
        lane_id_list = []
        for idx in oncoming_lanes:
            lane_id_list.append(idx.lane_id)

        if n_lane.lane_id not in lane_id_list:
            return ego_wps_des_speed
        # The aggressiveness is mapped from [0,10] to [0,0.8] domain which
        # represents the portion of intitial distantce which is used for
        # triggering the u-turn task.
        aggressiveness = 0.3 + 0.5 * self._agent_behavior.aggressiveness / 10
        distance_threshold = 8

        if not self._uturn_is_initialized:
            self._uturn_initial_distant = (
                -vehicle.pose.position[0] + neighborhood_vehicles[0].pose.position[0]
            )

            self._uturn_initial_velocity = neighborhood_vehicles[0].speed
            self._uturn_initial_height = 1 * (
                neighborhood_vehicles[0].pose.position[1] - vehicle.pose.position[1]
            )

            if (1 * self._uturn_initial_height * 3.14 / 13.8) * neighborhood_vehicles[
                0
            ].speed + distance_threshold > self._uturn_initial_distant:
                self._insufficient_initial_distant = True
            self._uturn_is_initialized = True

        horizontal_distant = (
            -vehicle.pose.position[0] + neighborhood_vehicles[0].pose.position[0]
        )
        vertical_distant = (
            neighborhood_vehicles[0].pose.position[1] - vehicle.pose.position[1]
        )

        if self._insufficient_initial_distant is True:
            if horizontal_distant > 0:
                return ego_wps_des_speed
            else:
                self._task_is_triggered = True

        if (
            horizontal_distant > 0
            and self._task_is_triggered is False
            and horizontal_distant
            > (1 - aggressiveness) * (self._uturn_initial_distant - 1)
            + aggressiveness
            * (
                (1 * self._uturn_initial_height * 3.14 / 13.8)
                * neighborhood_vehicles[0].speed
                + distance_threshold
            )
        ):
            return ego_wps_des_speed

        if not neighborhood_vehicles and not self._task_is_triggered:
            return ego_wps_des_speed

        lane = self._road_map.nearest_lane(pose.position)
        current_road = lane.road

        if self._task_is_triggered is False:
            self._uturn_initial_heading = pose.heading
            self._uturn_initial_position = pose.position[0]

        vehicle_heading_vec = radians_to_vec(pose.heading)
        initial_heading_vec = radians_to_vec(self._uturn_initial_heading)

        heading_diff = np.dot(vehicle_heading_vec, initial_heading_vec)

        lane = self._road_map.nearest_lane(Point(*vehicle.pose.position))
        speed_limit = lane.speed_limit / 1.5

        vehicle_dist = np.linalg.norm(
            vehicle.pose.position[:2] - neighborhood_vehicles[0].pose.position[:2]
        )
        if vehicle_dist < 5.5:
            speed_limit = 1.5 * lane.getSpeed()

        if (
            heading_diff < -0.95
            and pose.position[0] - self._uturn_initial_position < -2
        ):
            # Once it faces the opposite direction and pass the initial
            # uturn point for 2 meters, stop generating u-turn waypoints
            if (
                pose.position[0] - neighborhood_vehicles[0].pose.position[0] > 12
                or neighborhood_vehicles[0].pose.position[0] > pose.position[0]
            ):
                return ego_wps_des_speed
            else:
                speed_limit = neighborhood_vehicles[0].speed

        self._task_is_triggered = True

        target_lane_index = self._mission.task.target_lane_index
        target_lane_index = min(target_lane_index, len(oncoming_lanes) - 1)
        target_lane = oncoming_lanes[target_lane_index]

        offset = start_lane.offset_along_lane(Point(*pose.position))
        oncoming_offset = max(0, target_lane.length - offset)
        paths = self._paths_of_lane_at(target_lane, oncoming_offset, lookahead=30)

        target = paths[0][-1]

        heading = pose.heading
        target_heading = target.heading
        lane_width = target_lane.width
        lanes = len(current_road.lanes) + (len(oncoming_lanes) - target_lane_index)

        p0 = pose.position[:2]
        offset = radians_to_vec(heading) * lane_width
        p1 = np.array(
            [
                pose.position[0] + offset[0],
                pose.position[1] + offset[1],
            ]
        )
        offset = radians_to_vec(target_heading) * 5

        p3 = target.pos
        p2 = np.array([p3[0] - 5 * offset[0], p3[1] - 5 * offset[1]])

        p_x, p_y = bezier([p0, p1, p2, p3], 10)

        trajectory = []
        for i in range(len(p_x)):
            pos = np.array([p_x[i], p_y[i]])
            heading = Heading(vec_to_radians(target.pos - pos))
            lane = self._road_map.nearest_lane(Point(*pos))
            lane_id = lane.lane_id
            lane_index = lane_id.split("_")[-1]
            width = lane.width

            wp = Waypoint(
                pose=Pose(
                    position=pos, orientation=fast_quaternion_from_angle(heading)
                ),
                lane_width=width,
                speed_limit=speed_limit,
                lane_id=lane_id,
                lane_index=lane_index,
            )
            trajectory.append(wp)

        if self._first_uturn:
            uturn_activated_distance = math.sqrt(
                horizontal_distant ** 2 + vertical_distant ** 2
            )
            self._log.info(f"U-turn activated at distance: {uturn_activated_distance}")
            self._first_uturn = False

        return [trajectory]

    def _paths_of_lane_at(self, lane, offset, lookahead=30):
        wp_start = lane.from_lane_coord(RefLinePoint(offset))
        return self._waypoint_paths_on_lane_at(
            point=wp_start,
            lane_id=lane.getID(),
            lookahead=lookahead,
        )

    def _waypoint_paths_on_lane_at(
        self,
        point: Sequence,
        lane_id: str,
        lookahead: int,
        filter_edge_ids: Sequence[str] = None,
    ) -> List[List[Waypoint]]:
        """computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, constrained to filter_edge_ids if specified,
        starting at the nearest LanePoint to point within lane lane_id."""
        closest_linked_lp = self._lanepoints.closest_linked_lanepoint_on_lane_to_point(
            point, lane_id
        )
        return self._waypoints_starting_at_lanepoint(
            closest_linked_lp,
            lookahead,
            tuple(filter_edge_ids) if filter_edge_ids else (),
            tuple(point),
        )

    def _waypoint_paths_at(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: int = 5,
    ) -> List[List[Waypoint]]:
        closest_lane = self._road_map.nearest_lane(pose.position, radius=within_radius)
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += self._waypoint_paths_on_lane_at(
                pose.position, lane.lane_id, lookahead
            )
        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    def _waypoint_paths_along_route(
        self, point, lookahead: int, route
    ) -> List[List[Waypoint]]:
        """finds the closest lane to vehicle's position that is on its route,
        then gets waypoint paths from all lanes in its edge there."""
        assert len(route) > 0, f"Expected at least 1 edge in the route, got: {route}"
        closest_llp_on_each_route_edge = [
            self._lanepoints.closest_linked_lanepoint_on_edge(point, edge)
            for edge in route
        ]
        closest_linked_lp = min(
            closest_llp_on_each_route_edge,
            key=lambda l_lp: np.linalg.norm(
                vec_2d(l_lp.lp.pose.position) - vec_2d(point)
            ),
        )
        closest_lane = closest_linked_lp.lp.lane

        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += self._waypoint_paths_on_lane_at(
                point, lane.lane_id, lookahead, route
            )

        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    class _WaypointsCache:
        def __init__(self):
            self.lookahead = 0
            self.point = (0, 0, 0)
            self.filter_edge_ids = ()
            self._starts = {}

        def _match(self, lookahead, point, filter_edge_ids) -> bool:
            return (
                lookahead <= self.lookahead
                and point[0] == self.point[0]
                and point[1] == self.point[1]
                and filter_edge_ids == self.filter_edge_ids
            )

        def update(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_edge_ids: tuple,
            llp: LinkedLanePoint,
            paths: List[List[Waypoint]],
        ):
            if not self._match(lookahead, point, filter_edge_ids):
                self.lookahead = lookahead
                self.point = point
                self.filter_edge_ids = filter_edge_ids
                self._starts = {}
            self._starts[llp.lp.lane.index] = paths

        def query(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_edge_ids: tuple,
            llp: LinkedLanePoint,
        ) -> List[List[Waypoint]]:
            if self._match(lookahead, point, filter_edge_ids):
                hit = self._starts.get(llp.lp.lane.index, None)
                if hit:
                    # consider just returning all of them (not slicing)?
                    return [path[: (lookahead + 1)] for path in hit]
            return None

    def _waypoints_starting_at_lanepoint(
        self,
        lanepoint: LinkedLanePoint,
        lookahead: int,
        filter_edge_ids: tuple,
        point: Tuple[float, float, float],
    ) -> List[List[Waypoint]]:
        """computes equally-spaced Waypoints for all lane paths starting at lanepoint
        up to lookahead waypoints ahead, constrained to filter_edge_ids if specified."""

        # The following acts sort of like lru_cache(1), but it allows
        # for lookahead to be <= to the cached value...
        cache_paths = self._waypoints_cache.query(
            lookahead, point, filter_edge_ids, lanepoint
        )
        if cache_paths:
            return cache_paths

        lanepoint_paths = self._lanepoints.paths_starting_at_lanepoint(
            lanepoint, lookahead, filter_edge_ids
        )
        result = [
            SumoPlanner._equally_spaced_path(path, point) for path in lanepoint_paths
        ]

        self._waypoints_cache.update(
            lookahead, point, filter_edge_ids, lanepoint, result
        )

        return result

    @staticmethod
    def _equally_spaced_path(
        path: Sequence[LinkedLanePoint], point: Tuple[float, float, float]
    ) -> List[Waypoint]:
        """given a list of LanePoints starting near point, that may not be evenly spaced,
        returns the same number of Waypoints that are evenly spaced and start at point."""

        continuous_variables = [
            "positions_x",
            "positions_y",
            "headings",
            "lane_width",
            "speed_limit",
        ]
        discrete_variables = ["lane_id", "lane_index"]

        ref_lanepoints_coordinates = {
            parameter: [] for parameter in (continuous_variables + discrete_variables)
        }
        for idx, lanepoint in enumerate(path):
            if lanepoint.is_inferred and 0 < idx < len(path) - 1:
                continue
            ref_lanepoints_coordinates["positions_x"].append(
                lanepoint.lp.pose.position[0]
            )
            ref_lanepoints_coordinates["positions_y"].append(
                lanepoint.lp.pose.position[1]
            )
            ref_lanepoints_coordinates["headings"].append(
                lanepoint.lp.pose.heading.as_bullet
            )
            ref_lanepoints_coordinates["lane_id"].append(lanepoint.lp.lane.lane_id)
            ref_lanepoints_coordinates["lane_index"].append(lanepoint.lp.lane.index)
            ref_lanepoints_coordinates["lane_width"].append(lanepoint.lp.lane.width)
            ref_lanepoints_coordinates["speed_limit"].append(
                lanepoint.lp.lane.speed_limit
            )

        ref_lanepoints_coordinates["headings"] = inplace_unwrap(
            ref_lanepoints_coordinates["headings"]
        )
        first_lp_heading = ref_lanepoints_coordinates["headings"][0]
        lp_position = np.array([*path[0].lp.pose.position, 0])
        vehicle_pos = np.array([point[0], point[1], 0])
        heading_vector = np.array(
            [
                *radians_to_vec(first_lp_heading),
                0,
            ]
        )
        projected_distant_lp_vehicle = np.inner(
            (vehicle_pos - lp_position), heading_vector
        )

        ref_lanepoints_coordinates["positions_x"][0] = (
            lp_position[0] + projected_distant_lp_vehicle * heading_vector[0]
        )
        ref_lanepoints_coordinates["positions_y"][0] = (
            lp_position[1] + projected_distant_lp_vehicle * heading_vector[1]
        )
        # To ensure that the distance between waypoints are equal, we used
        # interpolation approach inspired by:
        # https://stackoverflow.com/a/51515357
        cumulative_path_dist = np.cumsum(
            np.sqrt(
                np.ediff1d(ref_lanepoints_coordinates["positions_x"], to_begin=0) ** 2
                + np.ediff1d(ref_lanepoints_coordinates["positions_y"], to_begin=0) ** 2
            )
        )

        if len(cumulative_path_dist) <= 1:
            return [Waypoint.from_LanePoint(path[0].lp)]

        evenly_spaced_cumulative_path_dist = np.linspace(
            0, cumulative_path_dist[-1], len(path)
        )

        evenly_spaced_coordinates = {}
        for variable in continuous_variables:
            evenly_spaced_coordinates[variable] = np.interp(
                evenly_spaced_cumulative_path_dist,
                cumulative_path_dist,
                ref_lanepoints_coordinates[variable],
            )

        for variable in discrete_variables:
            ref_coordinates = ref_lanepoints_coordinates[variable]
            evenly_spaced_coordinates[variable] = []
            jdx = 0
            for idx in range(len(path)):
                while (
                    jdx + 1 < len(cumulative_path_dist)
                    and evenly_spaced_cumulative_path_dist[idx]
                    > cumulative_path_dist[jdx + 1]
                ):
                    jdx += 1

                evenly_spaced_coordinates[variable].append(ref_coordinates[jdx])
            evenly_spaced_coordinates[variable].append(ref_coordinates[-1])

        equally_spaced_path = []
        for idx in range(len(path)):
            equally_spaced_path.append(
                Waypoint(
                    pose=Pose(
                        position=np.array(
                            [
                                evenly_spaced_coordinates["positions_x"][idx],
                                evenly_spaced_coordinates["positions_y"][idx],
                            ]
                        ),
                        orientation=fast_quaternion_from_angle(
                            Heading(evenly_spaced_coordinates["headings"][idx])
                        ),
                    ),
                    lane_width=evenly_spaced_coordinates["lane_width"][idx],
                    speed_limit=evenly_spaced_coordinates["speed_limit"][idx],
                    lane_id=evenly_spaced_coordinates["lane_id"][idx],
                    lane_index=evenly_spaced_coordinates["lane_index"][idx],
                )
            )

        return equally_spaced_path
