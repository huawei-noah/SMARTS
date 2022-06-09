from typing import Any, Dict

import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import Concatenate, FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.single_agent import SingleAgent


def make(config: Dict[str, Any]) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:multi_scenario-v0",
        scenario=config["scenario"],
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        wrappers=config["wrappers"],
        action_space=config["action_space"],
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env with action, reward, and observation wrapper
    env = intersection_info.Info(env=env)
    env = intersection_action.Action(env=env)
    env = intersection_reward.Reward(env=env)
    env = intersection_observation.Observation(env=env)

    # Check custom environment
    check_env(env)

    # Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
    env = VecMonitor(
        venv=env,
        filename=str(config["logdir"]),
        info_keywords=("is_success",),
    )

    return env


def make_all(config: Dict[str, Any]) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:multi-all-scenario-v0",
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        action_space=config["action_space"],
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env with action, reward, and observation wrapper
    env = FormatObs(env=env)
    env = FormatAction(env=env, space=ActionSpaceType[config["action_space"]])
    env = Info(env=env)
    env = Reward(env=env)
    env = DiscreteAction(env=env, space=config["action_wrapper"])
    env = FilterObs(env=env)

    # env = lambda env: FrameStack(env=env, num_stack=config["num_stack"])
    # env = Concatenate(env=env)

    env = SingleAgent(env=env)

    print("22222222222222222222222222222222222")
    print("obs space:", env.observation_space)
    print("act space:", env.action_space)
    print("333333333333333333333333333333333333")

    # Check custom environment
    check_env(env)

    print("1111111111111111111111111111111")

    # Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    print("111111111111111111111111111111122")

    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
    print("111111111111111111111111111111133")

    env = VecMonitor(
        venv=env,
        filename=str(config["logdir"]),
        info_keywords=("is_success",),
    )

    print("2222222222222222222222222222222222244")
    print("obs space:", env.observation_space)
    print("act space:", env.action_space)
    print("33333333333333333333333333333333333355")

    return env
