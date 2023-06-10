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

import logging
from functools import partial
from typing import Optional

from envision.client import Client as Envision
from envision.client import EnvisionDataFormatterArgs
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
    InterestDoneCriteria,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1, SumoOptions
from smarts.env.gymnasium.wrappers.limit_relative_target_pose import (
    LimitRelativeTargetPose,
)
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.env.utils.scenario import get_scenario_specs
from smarts.sstudio.scenario_construction import build_scenario

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

SUPPORTED_ACTION_TYPES = (
    ActionSpaceType.Continuous,
    ActionSpaceType.RelativeTargetPose,
)


def platoon_env(
    scenario: str,
    agent_interface: AgentInterface,
    seed: int = 42,
    headless: bool = True,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
):
    """All ego agents should track and follow the leader (i.e., lead vehicle) in a
    single-file fashion. The lead vehicle is marked as a vehicle of interest
    and may be found by filtering the
    :attr:`~smarts.core.observations.VehicleObservation.interest` attribute of
    the neighborhood vehicles in the observation. The episode ends when the
    leader reaches its destination. Ego agents do not have prior knowledge of the
    leader's destination.

    Observation space for each agent:
        Formatted :class:`~smarts.core.observations.Observation` using
        :attr:`~smarts.env.utils.observation_conversion.ObservationOptions.multi_agent`
        option is returned as observation. See
        :class:`~smarts.env.utils.observation_conversion.ObservationSpacesFormatter` for
        a sample formatted observation data structure.

    Action space for each agent:
        Action space for an ego can be either :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.Continuous`
        or :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.RelativeTargetPose`. User should choose
        one of the action spaces and specify the chosen action space through the ego's agent interface.

    Agent interface:
        Using the input argument agent_interface, users may configure any field of
        :class:`~smarts.core.agent_interface.AgentInterface`, except

        + :attr:`~smarts.core.agent_interface.AgentInterface.accelerometer`,
        + :attr:`~smarts.core.agent_interface.AgentInterface.done_criteria`,
        + :attr:`~smarts.core.agent_interface.AgentInterface.max_episode_steps`,
        + :attr:`~smarts.core.agent_interface.AgentInterface.neighborhood_vehicle_states`, and
        + :attr:`~smarts.core.agent_interface.AgentInterface.waypoint_paths`.

    Reward:
        Default reward is distance travelled (in meters) in each step, including the termination step.

    Episode termination:
        Episode is terminated if any of the following occurs.

        1. Lead vehicle reaches its pre-programmed destination.
        2. Steps per episode exceed 1000.
        3. Agent collides or drives off road.

    Args:
        scenario (str): Scenario name or path to scenario folder.
        agent_interface (AgentInterface): Agent interface specification.
        seed (int, optional): Random number generator seed. Defaults to 42.
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.

    Returns:
        An environment described by the input argument `scenario`.
    """

    # Check for supported action space
    assert agent_interface.action in SUPPORTED_ACTION_TYPES, (
        f"Got unsupported action type `{agent_interface.action}`, which is not "
        f"in supported action types `{SUPPORTED_ACTION_TYPES}`."
    )

    # Build scenario
    env_specs = get_scenario_specs(scenario)
    build_scenario(scenario=env_specs["scenario"])

    # Resolve agent interface
    resolved_agent_interface = resolve_agent_interface(agent_interface)
    agent_interfaces = {
        f"Agent_{i}": resolved_agent_interface for i in range(env_specs["num_agent"])
    }

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
        sim_name="VehicleFollowing",
        headless=headless,
        seed=seed,
        sumo_options=SumoOptions(headless=sumo_headless),
        visualization_client_builder=visualization_client_builder,
        observation_options=ObservationOptions.multi_agent,
    )
    if resolved_agent_interface.action == ActionSpaceType.RelativeTargetPose:
        env = LimitRelativeTargetPose(env)

    return env


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
        agents_alive=None,
        interest=InterestDoneCriteria(
            include_scenario_marked=True,
            strict=True,
        ),
    )
    max_episode_steps = 1000
    waypoints_lookahead = 80
    neighborhood_radius = 80
    return AgentInterface(
        accelerometer=True,
        action=agent_interface.action,
        done_criteria=done_criteria,
        drivable_area_grid_map=agent_interface.drivable_area_grid_map,
        lane_positions=agent_interface.lane_positions,
        lidar_point_cloud=agent_interface.lidar_point_cloud,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=neighborhood_radius),
        occupancy_grid_map=agent_interface.occupancy_grid_map,
        top_down_rgb=agent_interface.top_down_rgb,
        road_waypoints=agent_interface.road_waypoints,
        waypoint_paths=Waypoints(lookahead=waypoints_lookahead),
        signals=agent_interface.signals,
    )
