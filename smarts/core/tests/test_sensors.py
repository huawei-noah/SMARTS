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
import importlib.resources as pkg_resources
import math
import warnings
from functools import partial
from unittest import mock

import numpy as np
import pytest
from helpers.scenario import temp_scenario

from smarts.core import glsl
from smarts.core.agent_interface import CustomRenderCameraDependency
from smarts.core.coordinates import Heading, Pose, RefLinePoint
from smarts.core.observations import CustomRenderData
from smarts.core.plan import Plan
from smarts.core.scenario import Scenario
from smarts.core.sensor import (
    CustomRenderSensor,
    LanePositionSensor,
    RoadWaypointsSensor,
)
from smarts.core.sensors import (
    AccelerometerSensor,
    DrivenPathSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

AGENT_ID = "Agent-007"


def test_driven_path_sensor():
    vehicle = mock.Mock()
    sim = mock.Mock()

    max_path_length = 5
    sensor = DrivenPathSensor(max_path_length=max_path_length)

    positions = [(x, 0, 0) for x in range(0, 100, 10)]
    sim_times = list(range(0, 50, 5))
    for idx, (position, sim_time) in enumerate(zip(positions, sim_times)):
        sim.elapsed_sim_time = sim_time
        vehicle.pose.position = position
        sensor.track_latest_driven_path(sim.elapsed_sim_time, vehicle)

        if idx >= 3:
            assert sensor.distance_travelled(sim.elapsed_sim_time, last_n_steps=3) == 30
            assert (
                sensor.distance_travelled(sim.elapsed_sim_time, last_n_seconds=10) == 20
            )

        assert len(sensor()) <= max_path_length

    sensor.teardown()


@pytest.fixture
def scenario():
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

        return next(
            Scenario.variations_for_all_scenario_roots([str(scenario_root)], [AGENT_ID])
        )


def test_trip_meter_sensor(scenario):

    sim = mock.Mock()
    vehicle_state = mock.Mock()
    vehicle_state.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )
    vehicle_state.length = 3.68

    mission = scenario.missions[AGENT_ID]
    plan = Plan(scenario.road_map, mission)

    sensor = TripMeterSensor()
    waypoints_sensor = WaypointsSensor()

    positions = [(x, 0, 0) for x in range(0, 100, 10)]
    sim_times = list(range(0, 50, 5))
    for idx, (position, sim_time) in enumerate(zip(positions, sim_times)):
        sim.elapsed_sim_time = sim_time
        vehicle_state.position = position
        vehicle_state.pose = Pose(
            position=vehicle_state.position,
            orientation=np.array([0, 0, 0, 0]),
            heading_=Heading(0),
        )
        waypoint_paths = waypoints_sensor(vehicle_state, plan, scenario.road_map)
        sensor.update_distance_wps_record(
            waypoint_paths, vehicle_state, plan, sim.road_map
        )

    assert sensor() == sum(
        wpf.dist_to(wps.pos)
        for wpf, wps in zip(sensor._wps_for_distance[:-1], sensor._wps_for_distance[1:])
    )

    sensor.teardown()


def test_waypoints_sensor(scenario):
    vehicle_state = mock.Mock()
    vehicle_state.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )

    mission = scenario.missions[AGENT_ID]
    plan = Plan(scenario.road_map, mission)

    sensor = WaypointsSensor()
    waypoints = sensor(vehicle_state, plan, scenario.road_map)

    assert len(waypoints) == 3

    sensor.teardown()


def test_road_waypoints_sensor(scenario):
    assert isinstance(scenario, Scenario)
    assert scenario.road_map.map_spec
    assert scenario.road_map.map_spec.lanepoint_spacing == 1.0

    vehicle_state = mock.Mock()
    vehicle_state.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )

    mission = scenario.missions[AGENT_ID]
    plan = Plan(scenario.road_map, mission)

    sensor = RoadWaypointsSensor(4)
    road_waypoints = sensor(vehicle_state, plan, scenario.road_map)

    expected_lanes = {
        "edge-west-WE_0",
        "edge-west-WE_1",
        "edge-west-WE_2",
        "edge-west-EW_0",
        "edge-west-EW_1",
        "edge-west-EW_2",
    }

    assert len(road_waypoints) == 1

    lanes = road_waypoints[-1]
    assert len(expected_lanes ^ lanes.keys()) == 0
    assert len(lanes["edge-west-WE_0"][-1]) == 9

    sensor.teardown()


def test_accelerometer_sensor():
    dt = 0.1
    v_per_step = 15.0
    sensor = AccelerometerSensor()

    def _get_next(step, a_sensor):
        l_velocity = np.array((0, v_per_step * dt * step))
        a_velocity = np.array((0, v_per_step * dt * step))
        return sensor(l_velocity, a_velocity, dt=0.1)

    step = 1
    (linear_acc, angular_acc, linear_jerk, angular_jerk) = _get_next(step, sensor)

    assert np.all(linear_acc == np.array([0.0, 0.0, 0.0]))
    assert np.all(angular_acc == np.array([0.0, 0.0, 0.0]))
    assert np.all(linear_jerk == np.array([0.0, 0.0, 0.0]))
    assert np.all(angular_jerk == np.array([0.0, 0.0, 0.0]))

    step = 2
    (linear_acc, angular_acc, linear_jerk, angular_jerk) = _get_next(step, sensor)

    assert np.all(linear_acc == (0, v_per_step))
    assert np.all(angular_acc == (0, v_per_step))
    assert np.all(linear_jerk == np.array([0.0, 0.0, 0.0]))
    assert np.all(angular_jerk == np.array([0.0, 0.0, 0.0]))

    step = 3
    (linear_acc, angular_acc, linear_jerk, angular_jerk) = _get_next(step, sensor)

    assert np.all(linear_acc == (0, v_per_step))
    assert np.all(angular_acc == (0, v_per_step))
    assert np.all(linear_jerk == (0, 0))
    assert np.all(angular_jerk == (0, 0))

    sensor.teardown()


def test_lane_position_sensor(scenario):
    sensor = LanePositionSensor()

    vehicle_state = mock.Mock()
    vehicle_state.actor_id = "dummy"
    vehicle_state.pose = Pose(
        position=np.array([143.0, -11, 0.1]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )

    off = sensor(scenario.road_map.lane_by_id("edge-north-NS_0"), vehicle_state)

    assert off == RefLinePoint(20.999999999999996, 1.004987562112089, 0)
    sensor.teardown()


def test_signals_sensor():
    pass


def test_custom_render_sensor():
    from smarts.p3d.renderer import Renderer

    renderer = Renderer("R1")

    vehicle_state = mock.Mock()
    vehicle_state.actor_id = "dummy"
    vehicle_state.pose = Pose(
        position=np.array([33, -65, 0]),
        orientation=np.array([0, 0, 0, 0]),
        heading_=Heading(0),
    )

    with pkg_resources.path(glsl, "simplex.frag") as frag_shader:

        sensor_gen = partial(
            CustomRenderSensor,
            vehicle_state=vehicle_state,
            width=256,
            height=256,
            resolution=1,
            fragment_shader_path=frag_shader,
            renderer=renderer,
            render_dependencies=(),
            ogm_sensor=None,
            top_down_rgb_sensor=None,
            dagm_sensor=None,
        )
        sensor = sensor_gen(name="simplex")
        sensor2 = sensor_gen(
            name="2nd",
            render_dependencies=(
                CustomRenderCameraDependency(sensor.name, "iChannel0"),
            ),
        )

    renderer.render()

    image: CustomRenderData = sensor2(renderer)
    assert image.data.shape == (256, 256, 3)

    sensor.teardown()
    renderer.destroy()
