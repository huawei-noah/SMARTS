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
from unittest import mock

import numpy as np
import pytest
from helpers.scenario import temp_scenario

from smarts.core.coordinates import Heading, Pose
from smarts.core.plan import Plan
from smarts.core.scenario import Scenario
from smarts.core.sensors import DrivenPathSensor, TripMeterSensor, WaypointsSensor
from smarts.core.utils.math import squared_dist
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

AGENT_ID = "Agent-007"


def test_driven_path_sensor():
    vehicle = mock.Mock()
    sim = mock.Mock()

    max_path_length = 5
    sensor = DrivenPathSensor(vehicle, max_path_length=max_path_length)

    positions = [(x, 0, 0) for x in range(0, 100, 10)]
    sim_times = list(range(0, 50, 5))
    for idx, (position, sim_time) in enumerate(zip(positions, sim_times)):
        sim.elapsed_sim_time = sim_time
        vehicle.position = position
        sensor.track_latest_driven_path(sim)

        if idx >= 3:
            assert sensor.distance_travelled(sim, last_n_steps=3) == 30
            assert sensor.distance_travelled(sim, last_n_seconds=10) == 20

        assert len(sensor()) <= max_path_length

    sensor.teardown()


@pytest.fixture
def scenarios():
    with temp_scenario(name="straight", map="maps/6lane.net.xml") as scenario_root:
        ego_missions = [
            t.Mission(
                t.Route(
                    begin=("edge-west-WE", 0, 10),
                    end=("edge-east-WE", 0, "max"),
                )
            ),
        ]
        gen_scenario(
            t.Scenario(ego_missions=ego_missions),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )

def test_trip_meter_sensor(scenarios):
    scenario: Scenario = next(scenarios)

    sim = mock.Mock()
    vehicle = mock.Mock()
    vehicle.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )
    vehicle.length = 3.68

    mission = scenario.missions[AGENT_ID]
    plan = Plan(scenario.road_map, mission)

    sensor = TripMeterSensor(vehicle, scenario.road_map, plan)
    waypoints_sensor = WaypointsSensor(vehicle, plan)
    

    positions = [(x, 0, 0) for x in range(0, 100, 10)]
    sim_times = list(range(0, 50, 5))
    for idx, (position, sim_time) in enumerate(zip(positions, sim_times)):
        sim.elapsed_sim_time = sim_time
        vehicle.position = position
        vehicle.pose = Pose(
            position=vehicle.position,
            orientation=np.array([0, 0, 0, 0]),
            heading_=Heading(0),
        )
        waypoint_paths = waypoints_sensor()
        sensor.update_distance_wps_record(waypoint_paths=waypoint_paths)

    assert sensor() == sum(wpf.dist_to(wps.pos) for wpf, wps in zip(sensor._wps_for_distance[:-1], sensor._wps_for_distance[1:]))

    sensor.teardown()

def test_waypoints_sensor(scenarios):
    scenario = next(scenarios)
    sim = mock.Mock()
    vehicle = mock.Mock()
    vehicle.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )

    mission = scenario.missions[AGENT_ID]
    plan = Plan(scenario.road_map, mission)

    sensor = WaypointsSensor(vehicle, plan)
    waypoints = sensor()

    assert len(waypoints) == 3

    sensor.teardown()
