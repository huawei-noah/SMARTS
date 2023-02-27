import sys
from pathlib import Path

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from train.reward import Reward

# def wrappers(config: Dict[str, Any]):
#     Info,
#     lambda env: DummyVecEnv([lambda: env]),
#     lambda env: VecMonitor(venv=env, filename=str(config["logdir"]), info_keywords=("is_success",))

def make_env(env_id, scenario, agent_interface, config, seed):
    from smarts.env.gymnasium.wrappers.api_reversion import Api021Reversion
    from preprocess import Preprocess
    from smarts.env.wrappers.single_agent import SingleAgent

    env = gym.make(
        env_id,
        scenario=scenario,
        agent_interface=agent_interface,
        seed=seed,
        sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
        headless=not config.head,  # If False, enables Envision display.
    )
    env = Reward(env)
    env = Preprocess(env, config, agent_interface.top_down_rgb)
    env = Api021Reversion(env)
    env = SingleAgent(env),
    # env = DummyVecEnv([lambda: env]),
    # env = VecMonitor(venv=env, filename=str(config.logdir), info_keywords=("is_success",))
 
    return env
