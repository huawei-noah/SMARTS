# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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


import math

import numpy as np

import smarts.sstudio.sstypes as t
from smarts.core.coordinates import Dimensions, Heading, Point, RefLinePoint
from smarts.core.events import Events
from smarts.core.observations import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    RoadWaypoints,
    SignalObservation,
    TopDownRGB,
    VehicleObservation,
    ViaPoint,
    Vias,
)
from smarts.core.plan import EndlessGoal, NavigationMission, Start
from smarts.core.road_map import Waypoint
from smarts.core.signals import SignalLightState
from smarts.core.vehicle_state import Collision


def dummy_observation() -> Observation:
    """A dummy observation for tests and conversion."""
    return Observation(
        dt=0.1,
        step_count=1,
        elapsed_sim_time=0.2,
        events=Events(
            collisions=(Collision("v", "2"),),
            off_road=False,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            reached_goal=False,
            reached_max_episode_steps=False,
            agents_alive_done=False,
            interest_done=False,
        ),
        ego_vehicle_state=EgoVehicleObservation(
            id="AGENT-007-07a0ca6e",
            position=(161.23485529, 3.2, 0.0),
            bounding_box=Dimensions(length=3.68, width=1.47, height=1.0),
            heading=Heading(-1.5707963267948966),
            speed=5.0,
            steering=-0.0,
            yaw_rate=4.71238898038469,
            road_id="east",
            lane_id="east_2",
            lane_index=2,
            lane_position=RefLinePoint(161.23485529, 0.0, 0.0),
            mission=NavigationMission(
                start=Start(
                    position=np.array([163.07485529, 3.2]),
                    heading=Heading(-1.5707963267948966),
                    from_front_bumper=True,
                ),
                goal=EndlessGoal(),
                route_vias=(),
                start_time=0.1,
                entry_tactic=t.TrapEntryTactic(
                    start_time=0,
                    zone=None,
                    exclusion_prefixes=(),
                    default_entry_speed=None,
                ),
                via=(),
                vehicle_spec=None,
            ),
            linear_velocity=(5.000000e00, 3.061617e-16, 0.000000e00),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_acceleration=(0.0, 0.0, 0.0),
            linear_jerk=(0.0, 0.0, 0.0),
            angular_jerk=(0.0, 0.0, 0.0),
        ),
        under_this_agent_control=True,
        neighborhood_vehicle_states=(
            VehicleObservation(
                id="car-west_0_0-east_0_max-784511-726648-0-0.0",
                position=(-1.33354215, -3.2, 0.0),
                bounding_box=Dimensions(length=3.68, width=1.47, height=1.4),
                heading=Heading(-1.5707963267948966),
                speed=5.050372796758114,
                road_id="west",
                lane_id="west_0",
                lane_index=0,
                lane_position=RefLinePoint(-1.33354215, 0.0, 0.0),
            ),
            VehicleObservation(
                id="car-west_1_0-east_1_max--85270-726648-1-0.0",
                position=(-1.47159011, 0.0, 0.0),
                bounding_box=Dimensions(length=3.68, width=1.47, height=1.4),
                heading=Heading(-1.5707963267948966),
                speed=3.6410559446059954,
                road_id="west",
                lane_id="west_1",
                lane_index=1,
                lane_position=RefLinePoint(-1.47159011, 0.0, 0.0),
            ),
        ),
        waypoint_paths=[
            [
                Waypoint(
                    pos=np.array([192.00733923, -3.2]),
                    heading=Heading(-1.5707963267948966),
                    lane_id="east_0",
                    lane_width=3.2,
                    speed_limit=5.0,
                    lane_index=0,
                    lane_offset=192.00733923,
                ),
                Waypoint(
                    pos=np.array([193.0, -3.2]),
                    heading=Heading(-1.5707963267948966),
                    lane_id="east_0",
                    lane_width=3.2,
                    speed_limit=5.0,
                    lane_index=0,
                    lane_offset=193.0,
                ),
            ],
            [
                Waypoint(
                    pos=np.array([192.00733923, 0.0]),
                    heading=Heading(-1.5707963267948966),
                    lane_id="east_1",
                    lane_width=3.2,
                    speed_limit=5.0,
                    lane_index=1,
                    lane_offset=192.00733923,
                ),
                Waypoint(
                    pos=np.array([193.0, 0.0]),
                    heading=Heading(-1.5707963267948966),
                    lane_id="east_1",
                    lane_width=3.2,
                    speed_limit=5.0,
                    lane_index=1,
                    lane_offset=193.0,
                ),
            ],
        ],
        distance_travelled=0.0,
        lidar_point_cloud=(
            [
                np.array([1.56077973e02, 5.56008599e00, -7.24975635e-14]),
                np.array([math.inf, math.inf, math.inf]),
                np.array([math.inf, math.inf, math.inf]),
                np.array([1.66673185e02, 1.59127180e00, 9.07052211e-14]),
            ],
            [
                True,
                False,
                False,
                True,
            ],
            [
                (
                    np.array([161.23485529, 3.2, 1.0]),
                    np.array([143.32519217, 11.39649262, -2.47296355]),
                ),
                (
                    np.array([161.23485529, 3.2, 1.0]),
                    np.array([158.45533372, 22.69904572, -2.47296355]),
                ),
                (
                    np.array([161.23485529, 3.2, 1.0]),
                    np.array([176.14095458, 16.07426611, -2.47296355]),
                ),
                (
                    np.array([161.23485529, 3.2, 1.0]),
                    np.array([180.12197649, -2.38705439, -2.47296355]),
                ),
            ],
        ),
        drivable_area_grid_map=DrivableAreaGridMap(
            metadata=GridMapMetadata(
                resolution=0.1953125,
                width=256,
                height=256,
                camera_position=(161.235, 3.2, 73.6),
                camera_heading=-math.pi / 2,
            ),
            data=np.array(
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                dtype=np.uint8,
            ),
        ),
        occupancy_grid_map=OccupancyGridMap(
            metadata=GridMapMetadata(
                resolution=0.1953125,
                width=256,
                height=256,
                camera_position=(161.235, 3.2, 73.6),
                camera_heading=-math.pi / 2,
            ),
            data=np.array(
                [
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0], [0]],
                ],
                dtype=np.uint8,
            ),
        ),
        top_down_rgb=TopDownRGB(
            metadata=GridMapMetadata(
                resolution=0.1953125,
                width=256,
                height=256,
                camera_position=(161.235, 3.2, 73.6),
                camera_heading=-math.pi / 2,
            ),
            data=np.array(
                [
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                ],
                dtype=np.uint8,
            ),
        ),
        road_waypoints=RoadWaypoints(
            lanes={
                "east_0": [
                    [
                        Waypoint(
                            pos=np.array([180.00587138, -3.2]),
                            heading=Heading(-1.5707963267948966),
                            lane_id="east_0",
                            lane_width=3.2,
                            speed_limit=5.0,
                            lane_index=0,
                            lane_offset=180.00587138,
                        ),
                        Waypoint(
                            pos=np.array([181.0, -3.2]),
                            heading=Heading(-1.5707963267948966),
                            lane_id="east_0",
                            lane_width=3.2,
                            speed_limit=5.0,
                            lane_index=0,
                            lane_offset=181.0,
                        ),
                    ]
                ],
                "east_1": [
                    [
                        Waypoint(
                            pos=np.array([180.00587138, 0.0]),
                            heading=Heading(-1.5707963267948966),
                            lane_id="east_1",
                            lane_width=3.2,
                            speed_limit=5.0,
                            lane_index=1,
                            lane_offset=180.00587138,
                        ),
                        Waypoint(
                            pos=np.array([181.0, 0.0]),
                            heading=Heading(-1.5707963267948966),
                            lane_id="east_1",
                            lane_width=3.2,
                            speed_limit=5.0,
                            lane_index=1,
                            lane_offset=181.0,
                        ),
                    ]
                ],
            }
        ),
        signals=(
            SignalObservation(
                SignalLightState.GO, Point(181.0, 0.0), ("east_1",), None
            ),
        ),
        via_data=Vias(near_via_points=(ViaPoint((181.0, 0.0), 1, "east", 5.0, False),)),
        steps_completed=4,
    )
