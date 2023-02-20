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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import logging
import math
import os
import pathlib
from functools import partial
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from envision.client import Client as Envision
from envision.client import EnvisionDataFormatterArgs
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1, SumoOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenario
from smarts.core.agent_interface import AgentsAliveDoneCriteria, AgentsListAlive

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

SUPPORTED_ACTION_TYPES = (
    ActionSpaceType.ActuatorDynamic,
    ActionSpaceType.RelativeTargetPose,
)
MAXIMUM_SPEED_MPS = 28  # 28m/s = 100.8 km/h. This is a safe maximum speed.


def platoon_v0_env(
    scenario: str,
    agent_interface: AgentInterface,
    seed: int = 42,
    headless: bool = True,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
):
    """An environment with a mission to be completed by a single or multiple ego agents.

    Observation space for each agent:
        An unformatted :class:`~smarts.core.observations.Observation` is returned as observation.

    Action space for each agent:
        Action space for each agent is configured through its `AgentInterface`.
        The action space could be either of the following.

    Reward:
        Reward is distance travelled (in meters) in each step, including the termination step.

    Episode termination:
        Episode is terminated if any of the following occurs.

        1. Steps per episode exceed 800.
        2. Agent collides, drives off road, drives off route, or drives on wrong way.

    Args:
        scenario (str): Scenario name or path to scenario folder.
        agent_interface (AgentInterface): Agent interface specification.
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        seed (int, optional): Random number generator seed. Defaults to 42.
        visdom (bool, optional): If True, enables visualization of observed
            RGB images in Visdom. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.

    Returns:
        An environment described by the input argument `scenario`.
    """

    # Check for supported action space
    assert (
        agent_interface.action in SUPPORTED_ACTION_TYPES
    ), f"Got unsupported action type `{agent_interface.action}`, which is not " \
        f"in supported action types `{SUPPORTED_ACTION_TYPES}`."

    # Build scenario
    env_specs = _get_env_specs(scenario)
    build_scenario(scenario=env_specs["scenario"])

    # Resolve agent interface
    resolved_agent_interface = resolve_agent_interface(agent_interface)
    agent_interfaces = {
        f"Agent_{i}": resolved_agent_interface for i in range(env_specs["num_agent"])
    }

    # env_action_space = resolve_env_action_space(agent_interfaces)

    # Resolve action space
    # env_action_space = resolve_env_action_space(agent_interfaces)

    visualization_client_builder = None
    if not headless:
        visualization_client_builder = partial(
            Envision,
            endpoint=None,
            output_dir=envision_record_data_replay_path,
            headless=headless,
            data_formatter_args=EnvisionDataFormatterArgs(
                "base", enable_reduction=False
            ),
        )

    env = HiWayEnvV1(
        scenarios=[env_specs["scenario"]],
        agent_interfaces=agent_interfaces,
        sim_name="Platoon",
        headless=headless,
        visdom=visdom,
        seed=seed,
        sumo_options=SumoOptions(headless=sumo_headless),
        visualization_client_builder=visualization_client_builder,
        observation_options=ObservationOptions.multi_agent,
    )
    # env.action_space = env_action_space

    return env


def _get_env_specs(scenario: str):
    """Returns the appropriate environment parameters for each scenario.

    Args:
        scenario (str): Scenario

    Returns:
        Dict[str, Any]: A parameter dictionary.
    """

    if scenario == "1_to_2lane_left_turn_c":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_c"
            ),
            "num_agent": 1,
        }
    elif scenario == "1_to_2lane_left_turn_t":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_t"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_merge_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "merge"
                / "3lane_multi_agent"
            ),
            "num_agent": 2,
        }
    elif scenario == "3lane_merge_single_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "merge"
                / "3lane_single_agent"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_cruise_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "straight"
                / "3lane_cruise_multi_agent"
            ),
            "num_agent": 3,
        }
    elif os.path.isdir(scenario):
        import re

        regexp_agent = re.compile(r"agents_\d+")
        regexp_num = re.compile(r"\d+")
        matches_agent = regexp_agent.search(scenario)
        if not matches_agent:
            raise Exception(
                f"Scenario path should match regexp of 'agents_\\d+', but got {scenario}"
            )
        num_agent = regexp_num.search(matches_agent.group(0))

        return {
            "scenario": str(scenario),
            "num_agent": int(num_agent.group(0)),
        }
    else:
        raise Exception(f"Unknown scenario {scenario}.")


# def resolve_agent_action_space(agent_interface: AgentInterface):
#     """Get the competition action space for the given agent interface."""
#     assert (
#         agent_interface.action in SUPPORTED_ACTION_TYPES
#     ), f"Unsupported action type `{agent_interface.action}` not in supported actions `{SUPPORTED_ACTION_TYPES}`"

#     if agent_interface.action == ActionSpaceType.RelativeTargetPose:
#         max_dist = MAXIMUM_SPEED_MPS * 0.1  # assumes 0.1 timestep
#         return gym.spaces.Box(
#             low=np.array([-max_dist, -max_dist, -np.pi]),
#             high=np.array([max_dist, max_dist, np.pi]),
#             dtype=np.float32,
#         )
#     if agent_interface.action == ActionSpaceType.TargetPose:
#         return gym.spaces.Box(
#             low=np.array([-1e10, -1e10, -np.pi, 0.1]),
#             high=np.array([1e10, 1e10, np.pi, 0.1]),
#             dtype=np.float32,
#         )


# def resolve_env_action_space(agent_interfaces: Dict[str, AgentInterface]):
#     """Get the environment action space for the given set of agent interfaces."""
#     return gym.spaces.Dict(
#         {
#             a_id: resolve_agent_action_space(a_inter)
#             for a_id, a_inter in agent_interfaces.items()
#         }
#     )


def resolve_agent_interface(agent_interface: AgentInterface):
    """Resolve the agent interface for a given environment. Some interface
    values can be configured by the user, but others are pre-determined and
    fixed.
    """

    done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=False,
        on_shoulder=False,
        wrong_way=False,
        not_moving=False,
        # agents_alive=AgentsAliveDoneCriteria(
        #     agent_lists_alive=[
        #         AgentsListAlive(
        #             agents_list=[""]
        #         )
        #     ]
        # ),
    )
    max_episode_steps = 800
    road_waypoint_horizon = 50
    waypoints_lookahead = 50
    return AgentInterface(
        accelerometer=True,
        action=agent_interface.action,
        done_criteria=done_criteria,
        drivable_area_grid_map=agent_interface.drivable_area_grid_map,
        lidar_point_cloud=True,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicle_states=True,
        occupancy_grid_map=agent_interface.occupancy_grid_map,
        top_down_rgb=agent_interface.top_down_rgb,
        road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
        waypoint_paths=Waypoints(lookahead=waypoints_lookahead),
    )
