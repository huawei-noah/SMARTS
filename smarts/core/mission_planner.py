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
import random
import math
from typing import Optional

import numpy as np

from smarts.sstudio.types import MapZone, UTurn
from .sumo_road_network import SumoRoadNetwork
from .scenario import EndlessGoal, LapMission, Mission, Start, default_entry_tactic
from .waypoints import Waypoint, Waypoints
from .route import ShortestRoute, EmptyRoute
from .coordinates import Pose
from .utils.math import vec_to_radians, radians_to_vec, evaluate_bezier as bezier


class PlanningError(Exception):
    pass


class MissionPlanner:
    def __init__(self, waypoints: Waypoints, road_network: SumoRoadNetwork):
        self._waypoints = waypoints
        self._mission = None
        self._route = None
        self._road_network = road_network
        self._did_plan = False

    def random_endless_mission(
        self, min_range_along_lane=0.3, max_range_along_lane=0.9
    ):
        assert min_range_along_lane > 0  # Need to start further than beginning of lane
        assert max_range_along_lane < 1  # Cannot start past end of lane
        assert min_range_along_lane < max_range_along_lane  # Min must be less than max

        edge_id = self._road_network.random_route(1)[0]
        n_lane = random.choice(self._road_network.edge_by_id(edge_id).getLanes())

        # XXX: The ends of the edge are not as useful as starting mission locations.
        #      Sumo complains if we get too close to 0 or `lane_length`.
        offset = random.random() * min_range_along_lane + (
            max_range_along_lane - min_range_along_lane
        )
        offset *= n_lane.getLength()
        coord = self._road_network.world_coord_from_offset(n_lane, offset)
        nearest_wp = self._waypoints.closest_waypoint_on_lane(coord, n_lane.getID())
        return Mission(
            start=Start(tuple(nearest_wp.pos), nearest_wp.heading),
            goal=EndlessGoal(),
            entry_tactic=None,
        )

    def plan(self, mission: Optional[Mission]):
        self._mission = mission or self.random_endless_mission()

        if not self._mission.has_fixed_route:
            self._route = EmptyRoute()
        elif self._mission.task is not None:
            # TODO: ensure there is a default route
            self._route = EmptyRoute()
        else:
            start_lane = self._road_network.nearest_lane(
                self._mission.start.position,
                include_junctions=False,
                include_special=False,
            )
            start_edge = start_lane.getEdge()

            end_lane = self._road_network.nearest_lane(
                self._mission.goal.position,
                include_junctions=False,
                include_special=False,
            )
            end_edge = end_lane.getEdge()

            intermediary_edges = [
                self._road_network.edge_by_id(edge) for edge in self._mission.via
            ]

            self._route = ShortestRoute(
                self._road_network,
                edge_constraints=[start_edge] + intermediary_edges + [end_edge],
                wraps_around=isinstance(self._mission, LapMission),
            )

            assert not isinstance(self._route.edges, set), (
                "Cannot guarantee determinism of sumolib edge order if edges is a set"
                "Hashing order of `sumolib.net.edge` may be different across runs"
            )

            if len(self._route.edges) == 0:
                raise PlanningError(
                    "Unable to find a route between start={} and end={}. If either of "
                    "these are junctions (not well supported today) please switch to "
                    "edges and ensure there is a > 0 offset into the edge if it's "
                    "after a junction.".format(start_edge.getID(), end_edge.getID())
                )

        self._did_plan = True
        return self._mission

    @property
    def route(self):
        return self._route

    @property
    def mission(self):
        return self._mission

    def waypoint_paths_at(self, pose: Pose, lookahead: float):
        """Call assumes you're on the correct route already. We do not presently
        "replan" in case the route has changed.
        """
        assert (
            self._did_plan
        ), "Must call plan(...) before being able to invoke the mission planner."

        if self.mission.task is not None and isinstance(self.mission.task, UTurn):
            return self.uturn_waypoints(pose)

        else:
            edge_ids = self._edge_ids(pose)
            if edge_ids:
                return self._waypoints.waypoint_paths_along_route(
                    pose.position, lookahead, edge_ids
                )

            return self._waypoints.waypoint_paths_at(pose.position, lookahead)

    def waypoint_paths_on_lane_at(self, pose: Pose, lane_id: str, lookahead: float):
        """Call assumes you're on the correct route already. We do not presently
        "replan" in case the route has changed.
        """
        assert (
            self._did_plan
        ), "Must call plan(...) before being able to invoke the mission planner."

        edge_ids = self._edge_ids(pose, lane_id)
        if edge_ids:
            return self._waypoints.waypoint_paths_on_lane_at(
                pose.position, lane_id, lookahead, edge_ids
            )

        return self._waypoints.waypoint_paths_at(pose.position, lookahead)

    def _edge_ids(self, pose: Pose, lane_id: str = None):
        if self._mission.has_fixed_route:
            return [edge.getID() for edge in self._route.edges]

        # Filter waypoints to the internal lane the vehicle is driving on to deal w/
        # non-fixed routes (e.g. endless missions). This is so that the waypoints don't
        # jump between junction connections.
        if lane_id is None:
            # We take the 10 closest waypoints to then filter down to that which has
            # the closest heading. This way we get the waypoint on our lane instead of
            # a potentially closer lane that is on a different junction connection.
            closest_wps = self._waypoints.closest_waypoints(pose.position, 10)
            closest_wps = sorted(
                closest_wps, key=lambda wp: abs(pose.heading - wp.heading)
            )
            lane_id = closest_wps[0].lane_id

        edge = self._road_network.lane_by_id(lane_id).getEdge()
        if edge.getFunction() != "internal":
            return []

        edge_ids = [edge.getID()]
        next_edges = list(edge.getOutgoing().keys())

        assert (
            len(next_edges) <= 1
        ), "A junction is expected to have <= 1 outgoing edges"
        if next_edges:
            edge_ids.append(next_edges[0].getID())

        return edge_ids

    def uturn_waypoints(self, pose: Pose):
        # TODO: 1. Need to revisit the approach to calculate the U-Turn trajectory.
        #       2. Wrap this method in a helper.
        start_lane = self._road_network.nearest_lane(
            self._mission.start.position,
            include_junctions=False,
            include_special=False,
        )
        start_edge = self._road_network.road_edge_data_for_lane_id(start_lane.getID())
        wp = self._waypoints.closest_waypoint(pose.position)
        current_edge = self._road_network.edge_by_lane_id(wp.lane_id)
        if not start_edge.oncoming_edges:
            return []

        oncoming_edge = start_edge.oncoming_edges[0]
        oncoming_lanes = oncoming_edge.getLanes()
        target_lane_index = self._mission.task.target_lane_index
        target_lane_index = min(target_lane_index, len(oncoming_lanes) - 1)
        target_lane = oncoming_lanes[target_lane_index]

        offset = self._road_network.offset_into_lane(start_lane, pose.position[:2])
        oncoming_offset = max(0, target_lane.getLength() - offset)
        paths = self.paths_of_lane_at(target_lane, oncoming_offset, lookahead=30)
        target = paths[0][-1]

        heading = pose.heading
        target_heading = target.heading
        lane_width = target_lane.getWidth()
        lanes = (len(current_edge.getLanes())) + (
            len(oncoming_lanes) - target_lane_index
        )

        if current_edge.getID() != oncoming_edge.getID():
            # agent at the start edge
            p0 = pose.position[:2]
            distance = (
                10 * abs(abs(target_heading - heading) - math.pi / 2) / (math.pi / 2)
            )
            offset = radians_to_vec(heading) * distance
            p1 = np.array([pose.position[0] + offset[0], pose.position[1] + offset[1],])

            offset = radians_to_vec(heading + math.pi / 2) * (lane_width * lanes)
            p2 = np.array([p1[0] + offset[0], p1[1] + offset[1]])
            p3 = target.pos
            p_x, p_y = bezier([p0, p1, p2, p3], 20)
        else:
            # agent at the oncoming edge
            p0 = pose.position[:2]
            offset = radians_to_vec(heading) * lane_width
            p1 = np.array([pose.position[0] + offset[0], pose.position[1] + offset[1],])
            offset = radians_to_vec(target_heading) * 5
            p2 = np.array([p1[0] + offset[0], p1[1] + offset[1]])

            p3 = target.pos
            p_x, p_y = bezier([p0, p1, p2, p3], 20)

        trajectory = []
        for i in range(len(p_x)):
            pos = np.array([p_x[i], p_y[i]])
            heading = vec_to_radians(target.pos - pos)
            lane = self._road_network.nearest_lane(pos)
            lane_id = lane.getID()
            lane_index = lane_id.split("_")[-1]
            width = lane.getWidth()
            speed_limit = lane.getSpeed()

            wp = Waypoint(
                pos=pos,
                heading=heading,
                lane_width=width,
                speed_limit=speed_limit,
                lane_id=lane_id,
                lane_index=lane_index,
            )
            trajectory.append(wp)
        return [trajectory]

    def paths_of_lane_at(self, lane, offset, lookahead=30):
        wp_start = self._road_network.world_coord_from_offset(lane, offset)

        paths = self._waypoints.waypoint_paths_on_lane_at(
            point=wp_start, lane_id=lane.getID(), lookahead=lookahead,
        )
        return paths
