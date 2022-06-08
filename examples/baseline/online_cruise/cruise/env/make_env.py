from typing import Any, Dict

import gym
from cruise.env import action, reward
from stable_baselines3.common import monitor
from stable_baselines3.common.env_checker import check_env
from competition_env import CompetitionEnv

import smarts.env.wrappers.rgb_image as smarts_rgb_image
import smarts.env.wrappers.single_agent as smarts_single_agent
from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from  smarts.env.wrappers.format_obs import FormatObs


def make_env(config: Dict[str, Any]) -> gym.Env:

    vehicle_interface = smarts_agent_interface.AgentInterface(
        max_episode_steps=config["max_episode_steps"],
        rgb=smarts_agent_interface.RGB(
            width=config["rgb_pixels"],
            height=config["rgb_pixels"],
            resolution=config["rgb_meters"] / config["rgb_pixels"],
        ),
        action=getattr(
            smarts_controllers.ActionSpaceType,
            config["action_space_type"],
        ),
        done_criteria=smarts_agent_interface.DoneCriteria(
            collision=True,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=True,
        ),
    )

    agent_specs = {
        agent_id: smarts_agent.AgentSpec(
            interface=vehicle_interface,
            agent_builder=None,
        )
        for agent_id in config["agent_ids"]
    }

    scenarios = [
        str(config["scenarios_dir"].joinpath(scenario))
        for scenario in config["scenarios"]
    ]

    env = CompetitionEnv(scenarios=["scenarios/cruise"], max_episode_steps=300)

    # Wrap env with ActionWrapper
    env = action.Action(env=env)
    # Wrap env with RewardWrapper
    env = reward.Reward(env=env)
    # Wrap env with RGBImage wrapper to only get rgb images in observation
    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    # # Wrap env with SingleAgent wrapper to be Gym compliant
    env = smarts_single_agent.SingleAgent(env=env)
    env = monitor.Monitor(env=env)
    #check_env(env, warn=True)

    return env
