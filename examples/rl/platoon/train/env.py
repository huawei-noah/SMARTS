import sys
from pathlib import Path

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym

from smarts.zoo.agent_spec import AgentSpec


def make_env(env_id, scenario, agent_spec: AgentSpec, config, seed):
    from preprocess import Preprocess
    from stable_baselines3.common.monitor import Monitor
    from train.reward import Reward

    from smarts.env.gymnasium.wrappers.api_reversion import Api021Reversion
    from smarts.env.wrappers.single_agent import SingleAgent

    env = gym.make(
        env_id,
        scenario=scenario,
        agent_interface=agent_spec.interface,
        seed=seed,
        sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
        headless=not config.head,  # If False, enables Envision display.
    )
    env = Reward(env=env, crop=agent_spec.agent_params["crop"])
    env = Api021Reversion(env=env)
    env = SingleAgent(env=env)
    env = Preprocess(env=env, agent_spec=agent_spec)
    env = Monitor(env)

    return env
