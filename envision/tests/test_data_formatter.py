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
import os
from typing import Iterator
from unittest import mock
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

import smarts.sstudio.types as t
from envision.client import Client
from envision.data_formatter import (
    EnvisionDataFormatter,
    EnvisionDataFormatterArgs,
    Operation,
    _formatter_map,
)
from envision.types import State, TrafficActorState, TrafficActorType, VehicleType
from smarts.core import seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading
from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.bubbles import bubble_geometry
from smarts.core.tests.helpers.scenario import temp_scenario
from smarts.core.utils.file import unpack
from smarts.sstudio.genscenario import gen_scenario


@pytest.fixture
def covered_data():
    return [
        (VehicleType.Bus, [0]),
        (VehicleType.Car, [4]),
        (TrafficActorType.SocialVehicle, [0]),
        (TrafficActorType.Agent, [2]),
        (
            Waypoint(
                pos=np.array([4, 5, 3]),
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
                events=Events(
                    [], False, False, False, False, False, False, False, True
                ),
                driven_path=[(4, 4), (2, 2)],
                point_cloud=[[1, 3], [4, 2]],
                mission_route_geometry=[[(0, 2.2), (9, 4.4)], [(3.1, 42)]],
                waypoint_paths=[
                    [
                        Waypoint(
                            pos=np.array([4, 5, 3]),
                            heading=Heading(2.24),
                            lane_id="NE-NW",
                            lane_width=4,
                            speed_limit=10,
                            lane_index=0,
                        )
                    ],
                    [
                        Waypoint(
                            pos=np.array([9, 5, 3]),
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
                -3.14,  # heading
                20,  # speed
                (
                    [],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ),  # events
                [
                    [[4, 5, 3, 2.24, 1, 4, 10, 0]],
                    [[9, 5, 3, 1.11, 2, 2, 1.2, 1]],
                ],  # waypoints [p_x, p_y, p_z, heading, reduced_lane_id, lane_width, speed_limit, lane_index]
                [4, 4, 2, 2],  # driven path
                [1, 3, 4, 2],  # point cloud positions
                [[0, 2.2, 9, 4.4], [3.1, 42]],  # mission route geometry
                2,  # agent type
                0,  # vehicle type
                {0: "agent_007", 1: "NE-NW", 2: "NE-EW"},  # added values
                [],  # removed values
            ],
        ),
    ]


@pytest.fixture
def primitive_data():
    return [(1, [1]), (0.2, [0.2]), ("la", ["la"]), ([], [[]]), ((3, 5), [[3, 5]])]


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
                        events=Events(
                            [], False, False, False, False, False, False, False, True
                        ),
                    )
                    for i in range(2)
                },
                scenario_id="scene_id",
                scenario_name="scene_name",
                bubbles=[],
                scores=dict(),
                ego_agent_ids=["agent_007"],
                frame_time=0.1,
            ),
            [
                0.1,  # Timestamp
                "scene_id",  # scene_id
                "scene_name",  #
                [
                    [
                        0,
                        1,
                        4,
                        5,
                        2,
                        -3.14,
                        20,
                        [[], 0, 0, 0, 0, 0, 0, 0, 1],
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
                        -3.14,
                        20,
                        [[], 0, 0, 0, 0, 0, 0, 0, 1],
                        [],
                        [],
                        [],
                        [],
                        2,
                        0,
                    ],
                ],
                [],
                [],
                [0],
                # [], # ego agent ids
                {0: "agent_007", 1: "NE-NW"},  # lookup for reduced values
                [],
            ],
        ),
    ]


def test_covered_data_format(covered_data):
    for item in covered_data:
        es = EnvisionDataFormatter(EnvisionDataFormatterArgs(None))
        vt = item[0]
        _formatter_map[type(vt)](vt, es)

        data = es.resolve()

        assert data == item[1]
        assert data == unpack(data)


def test_primitive_data_format(primitive_data):
    for item in primitive_data:
        vt = item[0]
        es = EnvisionDataFormatter(EnvisionDataFormatterArgs(None))
        es.add_any(vt)

        data = es.resolve()

        assert data == item[1]
        assert data == unpack(data)


def test_layer():
    expected_output = [2, 5, 6, [2, 5, 6], [8, 8], ["Time", ["for", "tea", 12, "noon"]]]
    es = EnvisionDataFormatter(EnvisionDataFormatterArgs(None))
    es.add([2, 5, 6], op=Operation.FLATTEN)
    es.add([2, 5, 6])
    with es.layer():
        es.add([8, 8], op=Operation.FLATTEN)

    with es.layer():
        es.add("Time")
        with es.layer():
            es.add(["for", "tea", 12, "noon"], op=Operation.FLATTEN)

    data = es.resolve()

    assert data == expected_output
    assert data == unpack(data)


def test_complex_data(complex_data):
    for item in complex_data:
        vt = item[0]
        es = EnvisionDataFormatter(EnvisionDataFormatterArgs(None))
        es.add_any(vt)

        data = es.resolve()

        assert data == item[1]
        assert data == unpack(data)


@pytest.fixture
def sim_data():
    return [
        0.1,
        None,  # scene_id
        "straight",  # scene_name
        [
            [  # Traffic actor
                0,  # id
                1,  # lane id
                -1.84,  # x
                -3.2,  # y
                0.0,  # z
                -1.57,  # heading
                4.62,  # speed
                None,  # events
                [],  # wapypoint paths
                [],  # driven path
                [],  # lidar
                [],  # geometry
                0,  # actor type
                4,  # vehicle type
            ],
            [
                2,
                1,
                -1.84,
                0.0,
                0.0,
                -1.57,
                5.0,
                None,
                [],
                [],
                [],
                [],
                0,
                4,
            ],
            [
                3,
                1,
                -1.84,
                3.2,
                0.0,
                -1.57,
                4.91,
                None,
                [],
                [],
                [],
                [],
                0,
                4,
            ],
        ],
        [[90.0, -10.0, 90.0, 10.0, 110.0, 10.0, 110.0, -10.0, 90.0, -10.0]],  # bubbles
        [],
        [4],
        {
            0: "car-flow-route-west_0_0-east_0_max-7845114534199723832--7266489842092764092--0-0.0",
            1: None,
            2: "car-flow-route-west_1_0-east_1_max--852708111940723884--7266489842092764092--1-0.0",
            3: "car-flow-route-west_2_0-east_2_max--6324729949279915259--7266489842092764092--2-0.0",
            4: "AGENT_1",
        },
        [],
    ]


@pytest.fixture
def bubble():
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(20, 20)),
        margin=10,
        actor=t.BoidAgentActor(
            # TODO: Provide a more self-contained way to build agent locators for tests
            name="hive-mind",
            agent_locator="scenarios.straight.agent_prefabs:pose-boid-agent-v0",
        ),
    )


@pytest.fixture
def scenarios(bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("west", lane_idx, 0),
                        end=("east", lane_idx, "max"),
                    ),
                    rate=50,
                    actors={
                        t.TrafficActor("car"): 1,
                    },
                )
                for lane_idx in range(3)
            ]
        )

        ego_mission = t.Mission(t.Route(begin=("west", 0, 1), end=("east", 0, 1)))

        gen_scenario(
            t.Scenario(
                ego_missions=[ego_mission], traffic={"all": traffic}, bubbles=[bubble]
            ),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], ["AGENT_1"]
        )


@pytest.fixture
def smarts():
    laner = AgentInterface(
        max_episode_steps=1000,
        action=ActionSpaceType.Lane,
    )

    agents = {"AGENT_1": laner}
    with mock.patch(
        "envision.client.Client.headless", new_callable=PropertyMock
    ) as envision_headless:
        envision = Client(headless=True)
        envision_headless.return_value = False
        envision.read_and_send = MagicMock()
        # envision.teardown = MagicMock()
        smarts = SMARTS(
            agents,
            traffic_sim=SumoTrafficSimulation(),
            envision=envision,
        )
        yield smarts

        smarts.destroy()


def test_client_with_smarts(smarts: SMARTS, scenarios: Iterator[Scenario], sim_data):
    seed(int(os.getenv("PYTHONHASHSEED", 42)))

    envision = smarts.envision

    first_time = True

    def side_effect(state: State):
        nonlocal first_time
        if not first_time:
            return
        first_time = False
        es = EnvisionDataFormatter(EnvisionDataFormatterArgs(None))
        assert state.scenario_id is not None
        with mock.patch(
            "envision.types.State.scenario_id", new_callable=PropertyMock
        ) as state_scenario_id:
            state_scenario_id.return_value = None
            es.add_any(state)

            data = es.resolve()
            assert data == sim_data
            assert data == unpack(data)

    envision.send = MagicMock(side_effect=side_effect)
    scenario = next(scenarios)
    assert scenario.missions
    smarts.reset(scenario)
