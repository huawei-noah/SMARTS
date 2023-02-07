import sys
from pathlib import Path
from typing import Any, Dict

from smarts.core.agent import Agent
from smarts.zoo import registry

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))


def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    from action import Action as DiscreteAction
    from observation import Concatenate, FilterObs, SaveObs

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs
    from smarts.env.wrappers.frame_stack import FrameStack

    # fmt: off
    wrappers = [
        SaveObs,
        DiscreteAction,
        FilterObs,
        lambda env: FrameStack(env=env, num_stack=3),
        lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers


class ModelAgent(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        import network
        import stable_baselines3 as sb3lib

        model_path = Path(__file__).resolve().parents[0] / "saved_model.zip"
        self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Dict[str, Any], **kwargs):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): An observation from an ego agent step.

        Returns:
            Any: The action for the ego agent.
        """
        action, _ = self.model.predict(observation=obs, deterministic=True)

        return action


class RandomAgent(Agent):
    """A sample policy with random actions. Note that only the class named `Policy`
    will be tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym

        self._action_space = gym.spaces.Discrete(4)

    def act(self, obs: Dict[str, Any], **kwargs):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): An observation from an ego agent step.

        Returns:
            Any: The action for the ego agent.
        """
        return self._action_space.sample()


registry.register("competion-agent-v0", ModelAgent)
registry.register("random-agent-v0", RandomAgent)
