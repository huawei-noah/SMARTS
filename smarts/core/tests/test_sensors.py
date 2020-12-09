from unittest import mock

import numpy as np
import pytest

from helpers.scenario import temp_scenario
from smarts.core.coordinates import Heading, Pose
from smarts.core.mission_planner import MissionPlanner
from smarts.core.scenario import Scenario
from smarts.core.sensors import DrivenPathSensor
from smarts.core.sensors import WaypointsSensor
from smarts.core.waypoints import Waypoints
from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

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
                t.Route(begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, "max"),)
            ),
        ]
        gen_scenario(
            t.Scenario(ego_missions=ego_missions), output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


def test_waypoints_sensor(scenarios):
    scenario = next(scenarios)
    sim = mock.Mock()
    vehicle = mock.Mock()
    vehicle.pose = Pose(
        position=np.array([33, -65, 0]), orientation=[0, 0, 0, 0], heading_=Heading(0),
    )

    mission_planner = MissionPlanner(scenario.waypoints, scenario.road_network)
    mission = scenario.missions[AGENT_ID]
    mission_planner.plan(mission)

    sensor = WaypointsSensor(sim, vehicle, mission_planner)
    waypoints = sensor()

    assert len(waypoints) == 3


@pytest.fixture
def uturn_scenarios():
    with temp_scenario(name="straight", map="maps/6lane.net.xml") as scenario_root:
        ego_missions = [
            t.Mission(
                route=t.Route(
                    begin=("edge-west-WE", 0, 30), end=("edge-west-EW", 0, "max")
                ),
                task=t.UTurn(),
            ),
        ]
        gen_scenario(
            t.Scenario(ego_missions=ego_missions), output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


def test_waypoints_sensor_with_uturn_task(uturn_scenarios):
    scenario = next(uturn_scenarios)
    sim = mock.Mock()
    vehicle = mock.Mock()
    vehicle.pose = Pose(
        position=np.array([33, -65, 0]), orientation=[0, 0, 0, 0], heading_=Heading(0),
    )

    mission_planner = MissionPlanner(scenario.waypoints, scenario.road_network)
    mission = scenario.missions[AGENT_ID]
    mission_planner.plan(mission)
    mission_planner._task_is_triggered = True

    sensor = WaypointsSensor(sim, vehicle, mission_planner)
    waypoints = sensor()

    assert len(waypoints) == 1


@pytest.fixture
def cut_in_scenarios():
    with temp_scenario(name="straight", map="maps/6lane.net.xml") as scenario_root:
        ego_missions = [
            t.Mission(
                route=t.Route(
                    begin=("edge-west-WE", 0, 30), end=("edge-west-EW", 0, "max")
                ),
                task=t.CutIn(),
            ),
        ]
        gen_scenario(
            t.Scenario(ego_missions=ego_missions), output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


def test_waypoints_sensor_with_cut_in_task(cut_in_scenarios):
    scenario = next(cut_in_scenarios)

    sim = mock.Mock()
    nei_vehicle = mock.Mock()
    nei_vehicle.pose = Pose(
        position=np.array([25, -68, 0]), orientation=[0, 0, 0, 0], heading_=Heading(0),
    )
    sim.neighborhood_vehicles_around_vehicle = mock.MagicMock(
        return_value=[nei_vehicle]
    )

    vehicle = mock.Mock()
    vehicle.pose = Pose(
        position=np.array([35, -65, 0]), orientation=[0, 0, 0, 0], heading_=Heading(0),
    )

    mission_planner = MissionPlanner(scenario.waypoints, scenario.road_network)
    mission = scenario.missions[AGENT_ID]
    mission_planner.plan(mission)

    sensor = WaypointsSensor(sim, vehicle, mission_planner)
    waypoints = sensor()

    assert len(waypoints) == 1
