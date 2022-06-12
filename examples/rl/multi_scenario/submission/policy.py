from typing import Any, Dict

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from train.action import Action as DiscreteAction
from train.info import Info
from train.observation import FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.single_agent import SingleAgent


def submitted_wrappers(config: Dict[str, Any]):
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

def submitted_policy():
    policy = Policy()

    return policy


class Policy:
    def __init__():
        pass

    def act():
        pass