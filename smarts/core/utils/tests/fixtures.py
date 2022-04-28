# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import pytest

import smarts.sstudio.types as t
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Dimensions, Heading, RefLinePoint
from smarts.core.events import Events
from smarts.core.plan import EndlessGoal, Mission, Start
from smarts.core.road_map import Waypoint
from smarts.core.sensors import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    RoadWaypoints,
    TopDownRGB,
    VehicleObservation,
    Vias,
)


@pytest.fixture
def large_observation():
    return Observation(
        dt=0.1,
        step_count=1,
        elapsed_sim_time=0.2,
        events=Events(
            collisions=[],
            off_road=False,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            reached_goal=False,
            reached_max_episode_steps=False,
            agents_alive_done=False,
        ),
        ego_vehicle_state=EgoVehicleObservation(
            id="AGENT-007-07a0ca6e",
            position=np.array([161.23485529, 3.2, 0.0]),
            bounding_box=Dimensions(length=3.68, width=1.47, height=1.0),
            heading=Heading(-1.5707963267948966),
            speed=5.0,
            steering=-0.0,
            yaw_rate=4.71238898038469,
            road_id="east",
            lane_id="east_2",
            lane_index=2,
            lane_position=RefLinePoint(161.23485529, 0.0, 0.0),
            mission=Mission(
                start=Start(
                    position=np.array([163.07485529, 3.2]),
                    heading=Heading(-1.5707963267948966),
                    from_front_bumper=True,
                ),
                goal=EndlessGoal(),
                route_vias=(),
                start_time=0.1,
                entry_tactic=t.TrapEntryTactic(
                    wait_to_hijack_limit_s=0,
                    zone=None,
                    exclusion_prefixes=(),
                    default_entry_speed=None,
                ),
                via=(),
                vehicle_spec=None,
            ),
            linear_velocity=np.array([5.000000e00, 3.061617e-16, 0.000000e00]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            linear_acceleration=np.array([0.0, 0.0, 0.0]),
            angular_acceleration=np.array([0.0, 0.0, 0.0]),
            linear_jerk=np.array([0.0, 0.0, 0.0]),
            angular_jerk=np.array([0.0, 0.0, 0.0]),
        ),
        neighborhood_vehicle_states=[
            VehicleObservation(
                id="car-flow-route-west_0_0-east_0_max-7845114534199723832--7266489842092764092--0-0.0",
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
                id="car-flow-route-west_1_0-east_1_max--852708111940723884--7266489842092764092--1-0.0",
                position=(-1.47159011, 0.0, 0.0),
                bounding_box=Dimensions(length=3.68, width=1.47, height=1.4),
                heading=Heading(-1.5707963267948966),
                speed=3.6410559446059954,
                road_id="west",
                lane_id="west_1",
                lane_index=1,
                lane_position=RefLinePoint(-1.47159011, 0.0, 0.0),
            ),
        ],
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
                created_at=1649853761,
                resolution=0.1953125,
                width=256,
                height=256,
                camera_pos=(161.235, 3.2, 73.6),
                camera_heading_in_degrees=-90.0,
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
                created_at=1649853761,
                resolution=0.1953125,
                width=256,
                height=256,
                camera_pos=(161.235, 3.2, 73.6),
                camera_heading_in_degrees=-90.0,
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
                created_at=1649853761,
                resolution=0.1953125,
                width=256,
                height=256,
                camera_pos=(161.235, 3.2, 73.6),
                camera_heading_in_degrees=-90.0,
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
        via_data=Vias(near_via_points=[], hit_via_points=[]),
    )


@pytest.fixture
def adapter_data():
    return [
        (ActionSpaceType.ActuatorDynamic, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        (ActionSpaceType.Continuous, [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]),
        (ActionSpaceType.Lane, "keep_lane", "keep_lane"),
        (ActionSpaceType.LaneWithContinuousSpeed, [0, 20.2], [0, 20.2]),
        (
            ActionSpaceType.Trajectory,
            (
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
            ),
            (
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
            ),
        ),
        (
            ActionSpaceType.TrajectoryWithTime,
            [
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
                [0.1, 0.2],
            ],
            [
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
                [0.1, 0.2],
            ],
        ),
        (
            ActionSpaceType.MPC,
            [
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
            ],
            [
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
            ],
        ),
        (
            ActionSpaceType.TargetPose,
            (2, 4, -2.9, 20),
            (165.23485529, 1.2, 1.81238898, 20.0),
        ),
        (ActionSpaceType.Imitation, (2, 2), (2, 2)),
    ]
