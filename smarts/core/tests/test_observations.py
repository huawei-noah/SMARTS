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
import logging
import math
from typing import Dict

import gym
import numpy as np
import pytest
from panda3d.core import OrthographicLens, Point2, Point3

from smarts.core.agent import Agent
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DrivableAreaGridMap,
    NeighborhoodVehicles,
    RoadWaypoints,
    Signals,
)
from smarts.core.colors import SceneColors
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading, Point
from smarts.core.observations import DrivableAreaGridMap as ObsDrivableAreaGridMap
from smarts.core.observations import (
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    TopDownRGB,
)
from smarts.core.plan import Mission, PositionalGoal, Start, default_entry_tactic
from smarts.core.scenario import Scenario
from smarts.core.signals import SignalLightState
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"

NUM_STEPS = 30
MAP_WIDTH = 1536
MAP_HEIGHT = 1536
HALF_WIDTH = MAP_WIDTH / 2
HALF_HEIGHT = MAP_HEIGHT / 2
MAP_RESOLUTION = 50 / 256
SAMPLE_RANGE = 10

ROAD_COLOR = np.array(SceneColors.Road.value[0:3]) * 255


@pytest.fixture
def agent_interface():
    return AgentInterface(
        road_waypoints=RoadWaypoints(40),
        neighborhood_vehicle_states=NeighborhoodVehicles(
            radius=max(MAP_WIDTH * MAP_RESOLUTION, MAP_HEIGHT * MAP_RESOLUTION) * 0.5
        ),
        drivable_area_grid_map=DrivableAreaGridMap(
            width=MAP_WIDTH, height=MAP_HEIGHT, resolution=MAP_RESOLUTION
        ),
        occupancy_grid_map=OGM(
            width=MAP_WIDTH, height=MAP_HEIGHT, resolution=MAP_RESOLUTION
        ),
        top_down_rgb=RGB(width=MAP_WIDTH, height=MAP_HEIGHT, resolution=MAP_RESOLUTION),
        action=ActionSpaceType.Lane,
        signals=Signals(100.0),
    )


@pytest.fixture
def agent_spec(agent_interface):
    return AgentSpec(
        interface=agent_interface,
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )


@pytest.fixture
def env(agent_spec):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/sumo/figure_eight"],
        agent_specs={AGENT_ID: agent_spec},
        headless=True,
        visdom=False,
        fixed_timestep_sec=0.1,
        seed=42,
    )

    yield env
    env.close()


def project_2d(lens, img_metadata: GridMapMetadata, pos):
    center = np.array(img_metadata.camera_position)
    heading = np.radians(img_metadata.camera_heading_in_degrees)

    # Translate according to the camera center
    p_translated = pos - center

    # Apply the inverse rotation matrix to the vehicle position, same effect as rotating the camera
    p_rotated = np.array(
        [
            p_translated[0] * np.cos(-heading) - p_translated[1] * np.sin(-heading),
            p_translated[0] * np.sin(-heading) + p_translated[1] * np.cos(-heading),
            p_translated[2],
        ]
    )

    v_2d_pos_normalized = Point2()
    v_3d_pos = Point3(
        p_rotated[0], p_rotated[2], p_rotated[1]
    )  # y and z are flipped for project() in panda3D

    x = int(HALF_HEIGHT)
    y = int(HALF_WIDTH)

    # project() returns true if given 3d point is within the viewing frustum
    if lens.project(v_3d_pos, v_2d_pos_normalized):

        # v_2d_pos_normlized ranges (-1, 1) in both directions, with (-1,-1) being the lower-left corner
        # Sensor image has non-negative pixel positions, with (0, 0) starting from top-left corner
        # x and y are swapped between sensor image and the image projected from panda3D lens
        x = int(-v_2d_pos_normalized[1] * HALF_HEIGHT + HALF_HEIGHT)
        y = int(v_2d_pos_normalized[0] * HALF_WIDTH + HALF_WIDTH)

    return x, y


def apply_tolerance(arr, x, y, tolerance):
    return arr[x - tolerance : x + tolerance, y - tolerance : y + tolerance, :]


def sample_vehicle_pos(
    lens,
    rgb: TopDownRGB,
    ogm: OccupancyGridMap,
    drivable_area: ObsDrivableAreaGridMap,
    vehicle_pos,
):
    rgb_x, rgb_y = project_2d(lens, rgb.metadata, vehicle_pos)
    ogm_x, ogm_y = project_2d(lens, ogm.metadata, vehicle_pos)
    drivable_area_x, drivable_area_y = project_2d(
        lens, drivable_area.metadata, vehicle_pos
    )

    # Check if vehicles are rendered at the expected position
    # RGB
    tolerance = 2
    assert np.count_nonzero(rgb.data[rgb_x, rgb_y, :]) and np.count_nonzero(
        apply_tolerance(rgb.data, rgb_x, rgb_y, tolerance) != ROAD_COLOR
    )

    # OGM
    assert np.count_nonzero(apply_tolerance(ogm.data, ogm_x, ogm_y, tolerance))

    # Check if vehicles are within drivable area
    # Drivable area grid map
    assert np.count_nonzero(
        apply_tolerance(drivable_area.data, drivable_area_x, drivable_area_y, tolerance)
    )


def test_observations(env, agent_spec):
    agent = agent_spec.build_agent()
    observations: Dict[str, Observation] = env.reset()

    # Let the agent step for a while
    for _ in range(NUM_STEPS):
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, _, _, _ = env.step({AGENT_ID: agent_action})

    # RGB
    rgb = observations[AGENT_ID].top_down_rgb

    # OGM
    ogm = observations[AGENT_ID].occupancy_grid_map

    # Drivable area
    drivable_area = observations[AGENT_ID].drivable_area_grid_map

    lens = OrthographicLens()
    lens.setFilmSize(MAP_RESOLUTION * MAP_WIDTH, MAP_RESOLUTION * MAP_HEIGHT)

    # Check for ego vehicle
    ego_vehicle_position = observations[AGENT_ID].ego_vehicle_state.position
    sample_vehicle_pos(
        lens,
        rgb,
        ogm,
        drivable_area,
        ego_vehicle_position,
    )

    # Check for neighbor vehicles
    for neighbor_vehicle in observations[AGENT_ID].neighborhood_vehicle_states:
        sample_vehicle_pos(
            lens,
            rgb,
            ogm,
            drivable_area,
            neighbor_vehicle.position,
        )

    # Check for roads
    for paths in observations[AGENT_ID].road_waypoints.lanes.values():
        for path in paths:
            first_wp = path[0]
            last_wp = path[-1]

            first_wp_pos = np.array([first_wp.pos[0], first_wp.pos[1], 0])
            last_wp_pos = np.array([last_wp.pos[0], last_wp.pos[1], 0])

            rgb_first_wp_x, rgb_first_wp_y = project_2d(
                lens, rgb.metadata, first_wp_pos
            )
            rgb_last_wp_x, rgb_last_wp_y = project_2d(lens, rgb.metadata, last_wp_pos)

            drivable_area_first_wp_x, drivable_area_first_wp_y = project_2d(
                lens, drivable_area.metadata, first_wp_pos
            )
            drivable_area_last_wp_x, drivable_area_last_wp_y = project_2d(
                lens, drivable_area.metadata, last_wp_pos
            )

            # Check if roads are rendered at the start and end of road waypoint paths
            # RGB
            assert np.count_nonzero(rgb.data[rgb_first_wp_x, rgb_first_wp_y, :])
            assert np.count_nonzero(rgb.data[rgb_last_wp_x, rgb_last_wp_y, :])

            # Drivable Area Grid Map
            assert np.count_nonzero(
                drivable_area.data[
                    drivable_area_first_wp_x, drivable_area_first_wp_y, :
                ]
            )
            assert np.count_nonzero(
                drivable_area.data[drivable_area_last_wp_x, drivable_area_last_wp_y, :]
            )

    assert len(observations[AGENT_ID].signals) == 0


@pytest.fixture
def scenario():
    mission = Mission(
        start=Start(np.array((20.40, 68.40)), Heading(-0.5 * math.pi)),
        goal=PositionalGoal(Point(128.40, 0), 10),
        entry_tactic=default_entry_tactic(1.0),
    )
    return Scenario(
        scenario_root="scenarios/sumo/intersections/2lane",
        traffic_specs=[
            "scenarios/sumo/intersections/2lane/build/traffic/vertical.rou.xml"
        ],
        missions={AGENT_ID: mission},
    )


@pytest.fixture
def smarts(agent_interface):
    ai = agent_interface
    ai.action = ActionSpaceType.LaneWithContinuousSpeed
    smarts = SMARTS(
        {AGENT_ID: ai},
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )
    yield smarts
    smarts.destroy()


def test_signal_observations(smarts, scenario):
    observations: Dict[str, Observation] = smarts.reset(scenario)

    # go REAL SLOW so the light can change...
    agent = Agent.from_function(lambda _: (1.0, 0))

    # Let the agent step for a while
    for step in range(900):
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, _, dones, _ = smarts.step({AGENT_ID: agent_action})
        if dones[AGENT_ID]:
            break
        my_obs = observations[AGENT_ID]
        signals = my_obs.signals
        pos = my_obs.ego_vehicle_state.position
        if step < 2:
            # it's out of range...
            assert len(signals) == 0, f"step={step}, pos={pos}"
        else:
            assert len(signals) == 1, f"step={step}, pos={pos}"
            assert (
                signals[0].controlled_lanes[0] == ":junction-intersection_9_0"
            ), f"step={step}"
            assert len(signals[0].controlled_lanes) == 1, f"step={step}"
            if step < 449:
                assert signals[0].state == SignalLightState.STOP, f"{step}"
                assert signals[0].last_changed is None, f"{step}"
            elif step < 799:
                assert signals[0].state == SignalLightState.GO, f"{step}"
                assert np.isclose(signals[0].last_changed, 45.1), f"{step}"
            elif step < 899:
                assert signals[0].state == SignalLightState.CAUTION, f"{step}"
                assert np.isclose(signals[0].last_changed, 80.1), f"{step}"
            else:
                assert signals[0].state == SignalLightState.STOP, f"{step}"
                assert np.isclose(signals[0].last_changed, 90.1), f"{step}"
