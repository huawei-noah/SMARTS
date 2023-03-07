import sys
from pathlib import Path

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym


def make_env(env_id, scenario, agent_interface, config, seed):
    from preprocess import Preprocess
    from train.reward import Reward

    from smarts.env.gymnasium.wrappers.api_reversion import Api021Reversion
    from smarts.env.wrappers.single_agent import SingleAgent
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(
        env_id,
        scenario=scenario,
        agent_interface=agent_interface,
        seed=seed,
        sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
        headless=not config.head,  # If False, enables Envision display.
    )
    env = Reward(env)
    env = Api021Reversion(env)
    env = SingleAgent(env)
    env = Preprocess(env, config, agent_interface)
    env = Monitor(env)

    return env
