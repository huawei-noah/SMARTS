from typing import Any, Dict, List

import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.single_agent import SingleAgent


def wrappers(config: Dict[str, Any]):
    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType[config["action_space"]]),
        Info,
        Reward,
        lambda env: DiscreteAction(env=env, space=config["action_wrapper"]),
        FilterObs,
        SingleAgent,
        lambda env: DummyVecEnv([lambda: env]),
        lambda env: VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first"),
        lambda env: VecMonitor(venv=env, filename=str(config["logdir"]), info_keywords=("is_success",))
    ]
    # fmt: on

    return wrappers


def make(
    config: Dict[str, Any], scenario: str, wrappers: List[gym.Wrapper] = []
) -> gym.Env:
    """Make environment.

    Args:
        config (Dict[str, Any]): A dictionary of config parameters.
        scenario (str): Scenario
        wrappers (List[gym.Wrapper], optional): Sequence of gym environment wrappers.
            Defaults to empty list [].

    Returns:
        gym.Env: Environment corresponding to the `scenario`.
    """

    # Create environment
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=scenario,
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        wrappers=wrappers,
        action_space=config["action_space"],
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    # Check custom environment
    check_env(env)

    print("**************************************")
    print("obs space:", env.observation_space)
    print("act space:", env.action_space)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    return env


# def make_all(config: Dict[str, Any]) -> gym.Env:
#     # Create environment
#     env = gym.make(
#         "smarts.env:multi-all-scenario-v0",
#         img_meters=config["img_meters"],
#         img_pixels=config["img_pixels"],
#         action_space=config["action_space"],
#         headless=not config["head"],  # If False, enables Envision display.
#         visdom=config["visdom"],  # If True, enables Visdom display.
#         sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
#     )

#     # Wrap env
#     env = FormatObs(env=env)
#     env = FormatAction(env=env, space=ActionSpaceType[config["action_space"]])
#     env = Info(env=env)
#     env = Reward(env=env)
#     env = DiscreteAction(env=env, space=config["action_wrapper"])
#     env = FilterObs(env=env)
#     env = SingleAgent(env=env)

#     # Check custom environment
#     check_env(env)

#     # Wrap env with SB3 wrappers
#     env = DummyVecEnv([lambda: env])
#     env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
#     env = VecMonitor(
#         venv=env,
#         filename=str(config["logdir"]),
#         info_keywords=("is_success",),
#     )

#     # print("2222222222222222222222222222222222244")
#     # print("obs space:", env.observation_space)
#     # print("act space:", env.action_space)
#     # print("33333333333333333333333333333333333355")

#     return env
