from typing import Any, Dict, List

import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import Concatenate, FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.single_agent import SingleAgent


def wrappers(config: Dict[str, Any]):
    # fmt: off
    wrappers = [
        # Used to format observation space such that it becomes gym-space compliant.
        FormatObs,
        # Used to format action space such that it becomes gym-space compliant.
        lambda env: FormatAction(env=env, space=ActionSpaceType[config["action_space"]]),
        Info,
        # Used to shape rewards.
        Reward,
        # Used to discretize action space for easier RL training.
        lambda env: DiscreteAction(env=env, space=config["action_wrapper"]),
        # Used to filter only the selected observation parameters.
        FilterObs,
        # Used to stack sequential observations to include temporal information. 
        lambda env: FrameStack(env=env, num_stack=config["num_stack"]),
        # Concatenates stacked dictionaries into numpy arrays.
        lambda env: Concatenate(env=env, channels_order="first"),
        # Modifies interface to a single agent interface, which is compatible with libraries such as gym, Stable Baselines3, TF-Agents, etc.
        SingleAgent,
        lambda env: DummyVecEnv([lambda: env]),
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
        headless=True, # If False, enables Envision display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env
