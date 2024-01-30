# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, NewType, Optional, Tuple

import numpy as np

from smarts.core.coordinates import Heading, Point, RefLinePoint
from smarts.core.observations import Observation
from smarts.core.plan import NavigationMission, Plan, PositionalGoal, Start
from smarts.core.road_map import RoadMap
from smarts.core.utils.core_math import running_mean
from smarts.core.vehicle_index import VehicleIndex
from smarts.env.gymnasium.wrappers.metric.params import Params
from smarts.env.gymnasium.wrappers.metric.types import Costs
from smarts.env.gymnasium.wrappers.metric.utils import SlidingWindow, nearest_waypoint

Done = NewType("Done", bool)


def _collisions() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    sum = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal sum

        sum = sum + len(obs.events.collisions)
        j_coll = sum
        return Costs(collisions=j_coll)

    return func


def _comfort() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    jerk_linear_max = np.linalg.norm(np.array([0.9, 0.9, 0]))  # Units: m/s^3
    acc_linear_max = np.linalg.norm(np.array([2.0, 1.47, 0]))  # Units: m/s^2
    T_p = 30  # Penalty time steps = penalty time / delta time step = 3s / 0.1s = 30
    T_u = 0
    step = 0
    dyn_window = SlidingWindow(size=T_p)
    vehicle_pos = deque(maxlen=4)
    dt = 0.1
    min_disp = 0.1  # Minimum displacement to filter-out coordinate jitter. Units: m

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal jerk_linear_max, acc_linear_max, T_p, T_u, step, dyn_window, vehicle_pos, dt, min_disp

        step = step + 1
        vehicle_pos.appendleft(obs.ego_vehicle_state.position[:2])
        jerk = 0
        acc = 0
        if len(vehicle_pos) >= 3:
            disp_0 = np.linalg.norm(np.subtract(vehicle_pos[0], vehicle_pos[1]))
            disp_1 = np.linalg.norm(np.subtract(vehicle_pos[1], vehicle_pos[2]))
            speed_0 = disp_0 / dt
            speed_1 = disp_1 / dt
            if valid_0 := (disp_0 > min_disp and disp_1 > min_disp):
                acc = (speed_0 - speed_1) / dt
            if valid_0 and len(vehicle_pos) == 4:
                disp_2 = np.linalg.norm(np.subtract(vehicle_pos[2], vehicle_pos[3]))
                speed_2 = disp_2 / dt
                acc_1 = (speed_1 - speed_2) / dt
                if disp_2 > min_disp:
                    jerk = (acc - acc_1) / dt

        dyn = max(jerk / jerk_linear_max, acc / acc_linear_max)
        dyn_window.move(dyn)
        u_t = 1 if dyn_window.max() > 1 else 0
        T_u += u_t

        if not done:
            return Costs(comfort=np.nan)
        else:
            T_trv = step
            for _ in range(T_p):
                dyn_window.move(0)
                u_t = 1 if dyn_window.max() > 1 else 0
                T_u += u_t
            j_comfort = T_u / (T_trv + T_p)
            return Costs(comfort=j_comfort)

    return func


def _dist_to_destination(
    end_pos: Point,
    dist_tot: float,
    route: RoadMap.Route,
    prev_route_lane: RoadMap.Lane,
    prev_route_lane_point: Point,
    prev_route_displacement: float,
) -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0
    end_pos = end_pos
    dist_tot = dist_tot
    route = route
    prev_route_lane = prev_route_lane
    prev_route_lane_point = prev_route_lane_point
    prev_route_displacement = prev_route_displacement
    prev_dist_travelled = 0
    tot_dist_travelled = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step, end_pos, dist_tot, route, prev_route_lane, prev_route_lane_point, prev_route_displacement, prev_dist_travelled, tot_dist_travelled

        tot_dist_travelled += obs.distance_travelled

        if not done:
            cur_pos = Point(*obs.ego_vehicle_state.position)
            (
                cur_on_route,
                cur_route_lane,
                cur_route_lane_point,
                cur_route_displacement,
            ) = on_route(road_map=road_map, route=route, point=cur_pos)

            if cur_on_route:
                prev_route_lane = cur_route_lane
                prev_route_lane_point = cur_route_lane_point
                prev_route_displacement = cur_route_displacement
                prev_dist_travelled = tot_dist_travelled

            return Costs(dist_to_destination=np.nan)
        elif obs.events.reached_goal:
            return Costs(dist_to_destination=0)
        else:
            cur_pos = Point(*obs.ego_vehicle_state.position)
            (
                cur_on_route,
                cur_route_lane,
                cur_route_lane_point,
                cur_route_displacement,
            ) = on_route(road_map=road_map, route=route, point=cur_pos)

            # Step 1: Compute the last off-route distance driven by the vehicle, if any.
            if not cur_on_route:
                off_route_dist = tot_dist_travelled - prev_dist_travelled
                assert off_route_dist >= 0
                off_route_dist += prev_route_displacement
                last_route_lane = prev_route_lane
                last_route_pos = prev_route_lane_point
            else:
                off_route_dist = cur_route_displacement
                last_route_lane = cur_route_lane
                last_route_pos = cur_route_lane_point

            # Step 2: Compute the remaining route distance from the last recorded on-route position.
            on_route_dist = route.distance_between(
                start=RoadMap.Route.RoutePoint(pt=last_route_pos),
                end=RoadMap.Route.RoutePoint(pt=end_pos),
            )

            # Step 3: Compute absolute `on_route_dist` because it could be
            # negative when an agent overshoots the end position while
            # remaining outside the goal capture radius at all times.
            on_route_dist = abs(on_route_dist)

            # Step 4: Compute lane error penalty if vehicle is in the same road as goal, but in a different lane.
            # TODO: Lane error penalty should be computed. It is not computed
            # currently because the end lane of a SUMO traffic vehicle of
            # interest is currently not accessible.
            lane_error_dist = 0
            # end_lane = route.end_lane
            # if last_route_lane.road == end_lane.road:
            #     lane_error = abs(last_route_lane.index - end_lane.index)
            #     end_offset = end_lane.offset_along_lane(world_point=end_pos)
            #     lane_width, _ = end_lane.width_at_offset(end_offset)
            #     lane_error_dist = lane_error * lane_width

            # Step 5: Total distance to destination.
            dist_remainder = off_route_dist + on_route_dist + lane_error_dist

            # Step 6: Cap distance to destination.
            dist_remainder_capped = min(dist_remainder, dist_tot)

            return Costs(dist_to_destination=dist_remainder_capped / dist_tot)

    return func


def _dist_to_obstacles(
    ignore: List[str],
) -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0
    rel_angle_th = np.pi * 40 / 180
    rel_heading_th = np.pi * 179 / 180
    w_dist = 0.05
    safe_time = 3  # Safe driving distance expressed in time. Units:seconds.
    ignore = ignore

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step, rel_angle_th, rel_heading_th, w_dist, safe_time, ignore

        # Ego's position and heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = obs.ego_vehicle_state
        ego_heading = (ego.heading + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego.position

        # Set obstacle distance threshold using 3-second rule
        obstacle_dist_th = ego.speed * safe_time
        if obstacle_dist_th == 0:
            return Costs(dist_to_obstacles=0)

        # Get neighbors.
        nghbs = obs.neighborhood_vehicle_states

        # Filter neighbors by distance.
        nghbs_state = [
            (nghb, np.linalg.norm(np.subtract(nghb.position, ego_pos)))
            for nghb in nghbs
        ]
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if nghb_state[1] <= obstacle_dist_th
        ]
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # Filter neighbors to be ignored.
        nghbs_state = [
            nghb_state for nghb_state in nghbs_state if nghb_state[0].id not in ignore
        ]
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # Filter neighbors within ego's visual field.
        obstacles = []
        for nghb_state in nghbs_state:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            rel_pos = np.subtract(nghb_state[0].position, ego_pos)
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Relative angle is the angle rotation required by ego agent to face the obstacle.
            rel_angle = obstacle_angle - ego_heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) <= rel_angle_th:
                obstacles.append(nghb_state)
        nghbs_state = obstacles
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # Filter neighbors by their relative heading to that of ego's heading.
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if abs(nghb_state[0].heading.relative_to(ego.heading)) <= rel_heading_th
        ]
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # j_dist_to_obstacles : Distance to obstacles cost
        di = np.array([nghb_state[1] for nghb_state in nghbs_state])
        j_dist_to_obstacles = np.amax(np.exp(-w_dist * di))

        mean, step = running_mean(
            prev_mean=mean, prev_step=step, new_val=j_dist_to_obstacles
        )
        return Costs(dist_to_obstacles=mean)

    return func


def _jerk_linear() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0
    jerk_linear_max = np.linalg.norm(np.array([0.9, 0.9, 0]))  # Units: m/s^3
    """
    Maximum comfortable linear jerk as presented in:

    Bae, Il and et. al., "Self-Driving like a Human driver instead of a
    Robocar: Personalized comfortable driving experience for autonomous vehicles", 
    Machine Learning for Autonomous Driving Workshop at the 33rd Conference on 
    Neural Information Processing Systems, NeurIPS 2019, Vancouver, Canada.
    """

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step, jerk_linear_max

        jerk_linear = np.linalg.norm(obs.ego_vehicle_state.linear_jerk)
        j_l = min(jerk_linear / jerk_linear_max, 1)
        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_l)
        return Costs(jerk_linear=mean)

    return func


def _lane_center_offset() -> Callable[
    [RoadMap, VehicleIndex, Done, Observation], Costs
]:
    mean = 0
    step = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step

        if obs.events.off_road:
            # When vehicle is off road, the lane_center_offset cost
            # (i.e., j_lco) is set as zero.
            j_lco = 0
        else:
            # Vehicle's offset along the lane
            ego_pos = obs.ego_vehicle_state.position
            ego_lane = road_map.lane_by_id(obs.ego_vehicle_state.lane_id)
            reflinepoint = ego_lane.to_lane_coord(world_point=Point(*ego_pos))

            # Half width of lane
            lane_width, _ = ego_lane.width_at_offset(reflinepoint.s)
            lane_hwidth = lane_width * 0.5

            # Normalized vehicle's displacement from lane center
            # reflinepoint.t = signed distance from lane center
            norm_dist_from_center = reflinepoint.t / lane_hwidth

            # j_lco : Lane center offset
            j_lco = norm_dist_from_center**2

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_lco)
        return Costs(lane_center_offset=mean)

    return func


def _off_road() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    sum = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal sum

        sum = sum + obs.events.off_road
        j_off_road = sum
        return Costs(off_road=j_off_road)

    return func


def _speed_limit() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step

        if obs.events.off_road:
            # When vehicle is off road, the speed_limit cost (i.e., j_v) is
            # set as zero.
            j_speed_limit = 0
        else:
            # Nearest lane's speed limit.
            ego_speed = obs.ego_vehicle_state.speed
            ego_lane = road_map.lane_by_id(obs.ego_vehicle_state.lane_id)
            speed_limit = ego_lane.speed_limit
            assert speed_limit > 0, (
                "Expected lane speed limit to be a positive "
                f"float, but got speed_limit: {speed_limit}."
            )

            # Excess speed beyond speed limit.
            overspeed = ego_speed - speed_limit if ego_speed > speed_limit else 0
            overspeed_norm = min(overspeed / (0.5 * speed_limit), 1)
            j_speed_limit = overspeed_norm**2

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_speed_limit)
        return Costs(speed_limit=mean)

    return func


def _steps(
    max_episode_steps: int,
) -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    step = 0
    max_episode_steps = max_episode_steps

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal step, max_episode_steps

        step = step + 1

        if not done:
            return Costs(steps=np.nan)

        if obs.events.reached_goal or obs.events.interest_done:
            return Costs(steps=min(step, max_episode_steps) / max_episode_steps)
        elif (
            len(obs.events.collisions) > 0
            or obs.events.off_road
            or obs.events.reached_max_episode_steps
        ):
            return Costs(steps=1)
        else:
            raise CostError(
                "Expected reached_goal, collisions, off_road, "
                "max_episode_steps, or interest_done, to be true "
                f"on agent done, but got events: {obs.events}."
            )

    return func


def _vehicle_gap(
    num_agents: int,
    actor: str,
) -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0
    num_agents = num_agents
    aoi = actor  # Actor of interest, i.e., aoi
    vehicle_length = 4  # Units: m. Car length=3.68 m.
    safe_separation = 1  # Units: seconds. Minimum separation time between two vehicles.
    waypoint_spacing = 1  # Units: m. Assumption: Waypoints are spaced 1m apart.
    max_column_length = (num_agents + 1) * vehicle_length * 3.5  # Units: m.
    # Column length is the length of roadway a convoy of vehicles occupy,
    # measured from the lead vehicle (i.e., actor of interest) to the trail
    # vehicle. Here, num_agents is incremented by 1 to provide leeway for 1
    # traffic vehicle within the convoy. Multiplied by 3.5 to account for
    # vehicle separation distance while driving.
    min_waypoints_length = int(np.ceil(max_column_length / waypoint_spacing))

    def convert_2d_to_3d(array: np.ndarray) -> np.ndarray:
        return np.pad(array, (0, 1), mode="constant", constant_values=0)

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step, num_agents, aoi, vehicle_length, safe_separation, waypoint_spacing, max_column_length, min_waypoints_length

        if done == True:
            return Costs(vehicle_gap=mean)

        column_length = min(
            (
                num_agents * safe_separation * obs.ego_vehicle_state.speed
                + num_agents
                * vehicle_length
                * 2  # Multiplied by 2 to provide leeway when speed = 0.
            ),
            max_column_length,
        )

        # fmt: off
        waypoint_paths = obs.waypoint_paths
        max_len = max(map(len, waypoint_paths))
        mask = np.array([
            [[False, False, False]] * len(path) + [[True, True, True]] * (max_len - len(path))
            for path in waypoint_paths
        ])
        waypoints = np.array([
            list(map(lambda x: convert_2d_to_3d(x.pos), path)) + [np.full((3,), np.nan)] * (max_len - len(path))
            for path in waypoint_paths
        ])
        waypoints_masked = np.ma.MaskedArray(data=waypoints, mask=mask)
        # fmt: on

        # Find the nearest waypoint index to the actor of interest, if any.
        lane_width = waypoint_paths[0][0].lane_width
        aoi_pos = vehicle_index.vehicle_position(aoi)
        aoi_wp_ind, aoi_ind = nearest_waypoint(
            matrix=waypoints_masked,
            points=np.array([aoi_pos]),
            radius=lane_width,
        )

        if aoi_ind == None:
            # Actor of interest not found.
            j_gap = 1
        elif aoi_wp_ind[1] * waypoint_spacing > column_length:
            # Ego is outside of the maximum column length.
            j_gap = 1
        else:
            # Find the nearest waypoint index to the ego.
            ego_pos = obs.ego_vehicle_state.position
            dist = np.linalg.norm(waypoints_masked[:, 0, :] - ego_pos, axis=-1)
            ego_wp_inds = np.where(dist == dist.min())[0]

            if aoi_wp_ind[0] in ego_wp_inds:
                # Ego is in the same lane as the actor of interest.
                j_gap = max(aoi_wp_ind[1] * waypoint_spacing - vehicle_length, 0) / (
                    column_length - vehicle_length
                )
            else:
                # Ego is not in the same lane as the actor of interest.
                j_gap = 1

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_gap)
        return Costs(vehicle_gap=mean)

    return func


def _wrong_way() -> Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]:
    mean = 0
    step = 0

    def func(
        road_map: RoadMap, vehicle_index: VehicleIndex, done: Done, obs: Observation
    ) -> Costs:
        nonlocal mean, step
        j_wrong_way = 0
        if obs.events.wrong_way:
            j_wrong_way = 1

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_wrong_way)
        return Costs(wrong_way=mean)

    return func


@dataclass
class CostFuncsBase:
    """Functions to compute performance costs. Each cost function computes the
    running cost over time steps, for a given scenario."""

    # fmt: off
    collisions: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _collisions
    comfort: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _comfort
    dist_to_destination: Callable[[Point,float,RoadMap.Route,RoadMap.Lane,Point,float], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _dist_to_destination
    dist_to_obstacles: Callable[[List[str]], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _dist_to_obstacles
    jerk_linear: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _jerk_linear
    lane_center_offset: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _lane_center_offset
    off_road: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _off_road
    speed_limit: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _speed_limit
    steps: Callable[[int], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _steps
    vehicle_gap: Callable[[int, str], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _vehicle_gap
    wrong_way: Callable[[], Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]] = _wrong_way
    # fmt: on


CostFuncs = NewType(
    "CostFuncs", Dict[str, Callable[[RoadMap, VehicleIndex, Done, Observation], Costs]]
)


def make_cost_funcs(params: Params, **kwargs) -> CostFuncs:
    """
    Returns a dictionary of active cost functions to be computed as specified
    by the corresponding `active` field in `params`. Cost functions are
    initialized using `kwargs`, if any are provided.

    Args:
        params (Params): Parameters to configure individual cost functions.
        kwargs (Dict[str, Dict[str,Any]]): If any, used to initialize
            the appropriate cost functions.

    Returns:
        CostFuncs: Dictionary of active cost functions to be computed.
    """
    cost_funcs = CostFuncs({})
    for field in CostFuncsBase.__dataclass_fields__:
        if getattr(params, field).active:
            func = getattr(CostFuncsBase, field)
            args = kwargs.get(field, {})
            cost_funcs[field] = func(**args)

    return cost_funcs


class CostError(Exception):
    """Raised when computation of cost functions fail."""

    pass


def get_dist(
    road_map: RoadMap, point_a: Point, point_b: Point, tolerate: bool = False
) -> Tuple[float, RoadMap.Route]:
    """
    Computes the shortest route distance from point_a to point_b in the road
    map. Both points should lie on a road in the road map. Key assumption about
    the road map: Any two given points on the road map have valid routes in
    both directions.

    Args:
        road_map: Scenario road map.
        point_a: A point, in world-map coordinates, which lies on a road.
        point_b: A point, in world-map coordinates, which lies on a road.
        tolerate: If False, raises an error when distance is negative due to
            route being computed in reverse direction from point_b to point_a.
            Defaults to False.

    Returns:
        float: Shortest road distance between two points in the road map.
        RoadMap.Route: Planned route between point_a and point_b.
    """

    mission = NavigationMission(
        start=Start(
            position=point_a.as_np_array,
            heading=Heading(0),
            from_front_bumper=False,
        ),
        goal=PositionalGoal(
            position=point_b,
            radius=2,
        ),
    )
    plan = Plan(road_map=road_map, mission=mission, find_route=False)
    plan.create_route(mission=mission, start_lane_radius=3, end_lane_radius=0.5)
    assert isinstance(plan.route, RoadMap.Route)
    from_route_point = RoadMap.Route.RoutePoint(pt=point_a)
    to_route_point = RoadMap.Route.RoutePoint(pt=point_b)

    dist_tot = plan.route.distance_between(start=from_route_point, end=to_route_point)
    if dist_tot == None:
        raise CostError("Unable to find road on route near given points.")
    elif dist_tot < 0 and not tolerate:
        raise CostError(
            "Route computed in reverse direction from point_b to "
            f"point_a resulting in negative distance: {dist_tot}."
        )

    return dist_tot, plan.route


def on_route(
    road_map: RoadMap, route: RoadMap.Route, point: Point, radius: float = 7
) -> Tuple[bool, Optional[RoadMap.Lane], Optional[Point], Optional[float]]:
    """
    Computes whether `point` is within the search `radius` distance from any
    lane in the `route`.

    Args:
        road_map (RoadMap): Road map.
        route (RoadMap.Route): Route consisting of a set of roads.
        point (smarts.core.coordinates.Point): A world-coordinate point.
        radius (float): Search radius.

    Returns:
        Tuple[bool, Optional[RoadMap.Lane], Optional[smarts.core.coordinates.Point], Optional[float]]:
            True if `point` is nearby any road in `route`, else False. If true,
            additionally returns the (i) nearest lane in route, (ii) its
            nearest lane center point, and (iii) displacement between `point`
            and lane center point.
    """
    lanes = road_map.nearest_lanes(
        point=point,
        radius=radius,
        include_junctions=True,
    )

    route_roads = route.roads
    for lane, _ in lanes:
        if lane.road in route_roads:
            offset = lane.offset_along_lane(world_point=point)
            lane_point = lane.from_lane_coord(RefLinePoint(s=offset))
            displacement = np.linalg.norm(lane_point.as_np_array - point.as_np_array)
            return True, lane, lane_point, displacement

    return False, None, None, None
