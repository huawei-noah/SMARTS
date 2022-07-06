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
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.chassis import BoxChassis
from smarts.core.controllers.trajectory_interpolation_controller import (
    TrajectoryField,
    TrajectoryInterpolationController,
)
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.tests.helpers.scenario import temp_scenario
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.vehicle import Vehicle, VEHICLE_CONFIGS
from smarts.sstudio import gen_scenario
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "Agent-007"

# Tests are parameterized based on different trajectories
@pytest.fixture(
    params=[
        # Illegal inputs...
        {
            "error": np.array(
                [
                    [0.0],  # TIME
                    [100.0],  # X
                    [2.0],  # Y
                    [3.0],  # THETA
                    [4.0],  # VEL
                ]
            )
        },
        {
            "error": np.array(
                [
                    [0.0, 0.2],  # TIME
                    [12.0, 100.0],  # X
                    [2.0, np.nan],  # Y
                    [3.0, np.inf],  # THETA
                    [4.0, 400.0],  # VEL
                ]
            )
        },
        {
            "error": np.array(
                [
                    [1.0, 2.0, 3.0],  # TIME. Can not locate motion state.
                    [1.0, 2.0, 3.0],  # X
                    [2.0, 3.0, 4.0],  # Y
                    [3.0, 4.0, 5.0],  # THETA
                    [4.0, 5.0, 6.0],  # VEL
                ]
            )
        },
        # Trajectories with different time intervals...
        {
            "arb_interval": np.array(
                [
                    [0.0, 0.05, 0.2],  # TIME
                    [1.0, 2.0, 100.0],  # X
                    [0.0, 0.0, 0.0],  # Y
                    [0.0, 0.1, 3.0],  # THETA
                    [1.0, 1.0, 1.0],  # VEL
                ]
            ),
        },
        {
            "skipped_step": np.array(
                [
                    [0.0, 0.2, 0.3],  # TIME
                    [1.0, 20.0, 30.0],  # X
                    [0.0, 0.0, 0.0],  # Y
                    [0.0, 1.0, 2.0],  # THETA
                    [4.0, 5.0, 4.0],  # VEL
                ]
            ),
        },
        {
            "fine_grained_time": np.array(
                [
                    [0.0, 0.05, 0.1, 0.15, 0.2],  # TIME
                    [1.0, 2.0, 10.0, 200.0, 300.0],  # X
                    [0.0, 0.0, 0.0, 0.0, 0.0],  # Y
                    [0.0, 2.0, 3.0, 4.0, 5.0],  # THETA
                    [4.0, 4.0, 5.0, 5.0, 4.0],  # VEL
                ]
            ),
        },
        {
            "budda": np.array(
                [
                    [0.0, 0.15, 0.2],  # TIME
                    [1.0, 1.0, 10.0],  # X
                    [0.0, 0.0, 1.0],  # Y
                    [0.0, 0.0, 1.0],  # THETA
                    [4.0, 4.0, 5.0],  # VEL
                ]
            ),
        },
    ]
)
def controller_actions(request):
    return request.param


@pytest.fixture()
def scenario():
    with temp_scenario(name="map", map="maps/straight.net.xml") as scenario_root:
        mission = t.Mission(
            route=t.Route(
                begin=("west", 1, 99.9),
                end=("east", 1, 10.0),
            )
        )
        gen_scenario(
            t.Scenario(ego_missions=[mission]),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def bullet_client():
    client = bc.BulletClient(pybullet.DIRECT)
    yield client
    client.disconnect()


def test_trajectory_interpolation_controller(controller_actions, bullet_client):
    dt = 0.1
    i, j = np.ix_([TrajectoryField.X_INDEX, TrajectoryField.Y_INDEX], [0])

    for vehicle_id, trajectory in controller_actions.items():
        original_position = trajectory[i, j].reshape(2)
        original_heading = Heading(trajectory[TrajectoryField.THETA_INDEX][0])
        initial_speed = trajectory[TrajectoryField.VEL_INDEX][0]
        chassis = BoxChassis(
            pose=Pose.from_center(original_position, original_heading),
            speed=initial_speed,
            dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
            bullet_client=bullet_client,
        )
        vehicle = Vehicle(vehicle_id, chassis)

        has_error = False
        try:
            TrajectoryInterpolationController.perform_action(dt, vehicle, trajectory)
        except Exception:
            has_error = True

        new_pos = vehicle.pose.position[:2]

        if "error" in vehicle_id:
            assert has_error
        elif vehicle_id == "budda":
            assert np.linalg.norm(new_pos - original_position) < 1e-16
            assert np.isclose(vehicle.heading, original_heading)
        else:
            assert not np.linalg.norm(new_pos - original_position) < 1e-16
            assert not np.isclose(vehicle.heading, original_heading)


class WithTimeTrajectoryAgent(Agent):
    def act(self, obs):
        delta_traj = np.array(
            [
                [
                    0.0,
                    0.2,
                    0.3,
                ],  # TIME. Resolution is greater than SMARTS timestep.
                [1.0, 20.0, 30.0],  # X
                [0.0, 0.0, 0.0],  # Y
                [0.0, 1.0, 2.0],  # THETA
                [4.0, 5.0, 4.0],  # VEL
            ]
        )

        curr_position = obs.ego_vehicle_state.position
        curr_heading = obs.ego_vehicle_state.heading
        curr_speed = obs.ego_vehicle_state.speed
        new_origin_state = np.array(
            [
                [0.0],
                [curr_position[0]],
                [curr_position[1]],
                [curr_heading],
                [curr_speed],
            ]
        )

        return delta_traj + new_origin_state


@pytest.fixture
def agent_spec():
    return AgentSpec(
        interface=AgentInterface.from_type(AgentType.TrajectoryInterpolator),
        agent_builder=WithTimeTrajectoryAgent,
    )


@pytest.fixture
def smarts(agent_spec):
    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        fixed_timestep_sec=0.1,
    )
    yield smarts
    smarts.destroy()


def test_trajectory_interpolation_provider(smarts, agent_spec, scenario):
    """Test trajectory interpolation provider and controller.

    With different planning algorithm of WithTimeTrajectoryAgent,
    vehicle is going to accomplish its mission or not.

    """
    agent = agent_spec.build_agent()
    scenario = next(scenario)
    smarts.setup(scenario)
    observations = smarts.reset(scenario)
    init_ego_state = observations[AGENT_ID].ego_vehicle_state

    reached_goal = False
    for _ in range(5):
        agent_obs = observations[AGENT_ID]
        agent_action = agent.act(agent_obs)
        observations, _, dones, _ = smarts.step({AGENT_ID: agent_action})

        if agent_obs.events.reached_goal:
            reached_goal = True
            break

    curr_position = agent_obs.ego_vehicle_state.position
    curr_heading = agent_obs.ego_vehicle_state.heading
    curr_speed = agent_obs.ego_vehicle_state.speed

    init_position = init_ego_state.position
    init_heading = init_ego_state.heading
    init_speed = init_ego_state.speed
    assert np.linalg.norm(curr_position[:2] - init_position[:2]) > 1e-16
    assert not np.isclose(curr_heading, init_heading)
    assert not np.isclose(curr_speed, init_speed)
