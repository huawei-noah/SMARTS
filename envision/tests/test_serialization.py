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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Sequence

import pytest

from envision.serialization import Context, EnvisionDataFormatter, _serialization_map
from envision.types import State, TrafficActorState, TrafficActorType, VehicleType
from smarts.core.coordinates import Heading
from smarts.core.events import Events
from smarts.core.road_map import Waypoint


@pytest.fixture
def covered_data():
    return [
        (VehicleType.Bus, [0]),
        (VehicleType.Car, [4]),
        (TrafficActorType.SocialVehicle, [0]),
        (TrafficActorType.Agent, [2]),
        (
            Waypoint(
                pos=[4, 5, 3],
                heading=Heading(2.24),
                lane_id="NE-NW",
                lane_width=4,
                speed_limit=10,
                lane_index=0,
            ),
            [4, 5, 3, 2.24, 0, 4, 10, 0, {0: "NE-NW"}, []],
        ),
        (
            TrafficActorState(
                actor_type=TrafficActorType.Agent,
                vehicle_type=VehicleType.Bus,
                position=(4, 5, 2),
                heading=-3.141571,
                speed=20,
                name="",
                actor_id="agent_007",
                lane_id="NE-NW",
                score=10,
                events=Events(
                    [], False, False, False, False, False, False, False, True
                ),
                driven_path=[[4, 4], [2, 2]],
                point_cloud=[[1, 3], [4, 2]],
                mission_route_geometry=[[[0, 2.2], [9, 4.4]], [[3.1, 42]]],
                waypoint_paths=[
                    [
                        Waypoint(
                            pos=[4, 5, 3],
                            heading=Heading(2.24),
                            lane_id="NE-NW",
                            lane_width=4,
                            speed_limit=10,
                            lane_index=0,
                        )
                    ],
                    [
                        Waypoint(
                            pos=[9, 5, 3],
                            heading=Heading(1.11),
                            lane_id="NE-EW",
                            lane_width=2,
                            speed_limit=1.2,
                            lane_index=1,
                        )
                    ],
                ],
            ),
            [
                0,  # actor id lookup into values
                1,  # lane id lookup into values
                4,  # x
                5,  # y
                2,  # z
                -3.141571,  # heading
                20,  # speed
                (
                    [],
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ),  # events
                10,  # score
                [
                    [[4, 5, 3, 2.24, 1, 4, 10, 0]],
                    [[9, 5, 3, 1.11, 2, 2, 1.2, 1]],
                ],  # waypoints [p_x, p_y, p_z, heading, reduced_lane_id, lane_width, speed_limit, lane_index]
                [4, 4, 2, 2],  # driven path
                [1, 3, 4, 2],  # point cloud positions
                [[0, 2.2, 9, 4.4], [3.1, 42]],  # mission route geometry
                2,
                0,
                {0: "agent_007", 1: "NE-NW", 2: "NE-EW"},  # added values
                [],  # removed values
            ],
        ),
    ]


@pytest.fixture
def primitive_data():
    return [(1, [1]), (0.2, [0.2]), ("la", ["la"]), ([], [[]]), ((3, 5), [(3, 5)])]


@pytest.fixture
def complex_data():
    return [
        (
            State(
                traffic={
                    f"agent_00{i}": TrafficActorState(
                        actor_type=TrafficActorType.Agent,
                        vehicle_type=VehicleType.Bus,
                        position=(4, 5, 2),
                        heading=-3.141571,
                        speed=20,
                        name="",
                        actor_id="agent_007",
                        lane_id="NE-NW",
                        score=10,
                        events=Events(
                            [], False, False, False, False, False, False, False, True
                        ),
                    )
                    for i in range(2)
                },
                scenario_id="scene_id",
                scenario_name="big blue",
                bubbles=[],
                scene_colors=[],
                scores=[],
                ego_agent_ids=[],
                position=[],
                speed=[],
                heading=[],
                lane_ids=[],
                frame_time=0.1,
            ),
            [
                0.1,
                "scene_id",
                "big blue",
                [
                    [
                        0,
                        1,
                        4,
                        5,
                        2,
                        -3.141571,
                        20,
                        ([], False, False, False, False, False, False, False, True),
                        10,
                        [],
                        [],
                        [],
                        [],
                        2,
                        0,
                    ],
                    [
                        0,
                        1,
                        4,
                        5,
                        2,
                        -3.141571,
                        20,
                        ([], False, False, False, False, False, False, False, True),
                        10,
                        [],
                        [],
                        [],
                        [],
                        2,
                        0,
                    ],
                ],
                [],
                # [], # ego agent ids
                {0: "agent_007", 1: "NE-NW"},  # lookup for reduced values
                [],
            ],
        ),
    ]


def test_covered_data_format(covered_data):
    for item in covered_data:
        es = EnvisionDataFormatter(None)
        vt = item[0]
        _serialization_map[type(vt)](vt, es)

        data = es.resolve()

        assert data == item[1]


def test_primitive_data_format(primitive_data):
    for item in primitive_data:
        vt = item[0]
        es = EnvisionDataFormatter(None)
        es.add_any(vt)

        data = es.resolve()

        assert data == item[1]


def test_layer():
    expected_output = [2, 5, 6, [2, 5, 6], [8, 8], ["Time", ["for", "tea", 12, "noon"]]]
    es = EnvisionDataFormatter(None)
    es.add([2, 5, 6], "", op=Context.FLATTEN)
    es.add([2, 5, 6], "")
    with es.layer():
        es.add([8, 8], "", op=Context.FLATTEN)

    with es.layer():
        es.add("Time", "")
        with es.layer():
            es.add(["for", "tea", 12, "noon"], "", op=Context.FLATTEN)

    assert es.resolve() == expected_output


def test_complex_data(complex_data):
    for item in complex_data:
        vt = item[0]
        es = EnvisionDataFormatter(None)
        es.add_any(vt)

        data = es.resolve()

        assert data == item[1]
