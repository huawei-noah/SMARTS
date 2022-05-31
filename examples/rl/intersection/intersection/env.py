from typing import Any, Dict

import gym
from intersection import action as intersection_action
from intersection import info as intersection_info
from intersection import observation as intersection_observation
from intersection import reward as intersection_reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor


def make(config: Dict[str, Any]) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
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
