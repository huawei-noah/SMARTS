import sys
from pathlib import Path
from typing import Any, Dict, List

import gym
import gymnasium
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train.info import Info
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.wrappers.api_reversion import Api021Reversion
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.single_agent import SingleAgent

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submission.action import Action as DiscreteAction
from submission.observation import Concatenate, FilterObs, SaveObs


def wrappers(config: Dict[str, Any]):
    # fmt: off
    wrappers_ = [
        Info,
        # Used to shape rewards.
        Reward,
        # Used to save selected observation parameters for use in DiscreteAction wrapper.
        SaveObs,
        # Used to discretize action space for easier RL training.
        DiscreteAction,
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

    return wrappers_


def make(
    config: Dict[str, Any], scenario: str, wrappers: List[gym.Wrapper] = None
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
    wrappers = wrappers if wrappers is not None else []
    # Create environment
    env = gymnasium.make(
        "smarts.env.gymnasium:driving-smarts-competition-v0",
        scenario=scenario,
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
        headless=not config["head"],  # If False, enables Envision display.
    )
    # necessary to use the current stable baselines 3 by reverting to the 0.21 gym interface.
    env = Api021Reversion(env)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env
