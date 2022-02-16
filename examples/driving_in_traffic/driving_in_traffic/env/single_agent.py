from functools import partial
from typing import Any, Callable, Dict, Generator

import gym
from driving_in_traffic.env import action, reward

from smarts.core import agent as smarts_agent
from smarts.core import agent_interface as smarts_agent_interface
from smarts.core import controllers as smarts_controllers
from smarts.env import hiway_env as smarts_hiway_env
from smarts.env.wrappers import rgb_image as smarts_rgb_image
from smarts.env.wrappers import single_agent as smarts_single_agent


def gen_env(
    config: Dict[str, Any], seed: int
) -> Generator[Callable[[str], gym.Env], None, None]:
    base_seed = seed
    while True:
        yield partial(make_env, config=config, seed=base_seed)
        base_seed += 1


def make_env(config: Dict[str, Any], seed: int, env_name: str = None) -> gym.Env:

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
            not_moving=False,
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

    env = smarts_hiway_env.HiWayEnv(
        scenarios=scenarios,
        agent_specs=agent_specs,
        headless=config["headless"],
        visdom=config["visdom"],
        seed=seed,
        sim_name=env_name,
    )

    # Wrap env with ActionWrapper
    env = action.Action(env=env)
    # Wrap env with RewardWrapper
    env = reward.Reward(env=env)
    # Wrap env with RGBImage wrapper to only get rgb images in observation
    env = smarts_rgb_image.RGBImage(env=env, num_stack=1)
    # Wrap env with SingleAgent wrapper to be Gym compliant
    env = smarts_single_agent.SingleAgent(env=env)

    return env
