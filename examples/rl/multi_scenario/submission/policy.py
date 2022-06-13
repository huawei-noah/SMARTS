from typing import Any, Dict

from train.action import Action as DiscreteAction
from train.observation import Concatenate, FilterObs
from train.reward import Reward

from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.env.wrappers.frame_stack import FrameStack

# Environment variables (optional)
IMG_METERS=50 # Observation image area size in meters.
IMG_PIXELS=112 # Observation image size in pixels.


def submitted_wrappers(config: Dict[str, Any]):
    """Return environment wrappers for wrapping the evaluation environment. Use
    of wrappers is optional. If wrappers are not used, return empty list [].

    Args:
        config (Dict[str, Any]): _description_

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType[config["action_space"]]),
        Reward,
        lambda env: DiscreteAction(env=env, space=config["action_wrapper"]),
        FilterObs,
        lambda env: FrameStack(env=env, num_stack=config["num_stack"]),
        lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers


class Policy:
    """Policy class to be submitted.
    """
    def __init__(self):
        """All policy initialization matters, including loading of model, is 
        performed here. To be implemented by the user.
        """

        import stable_baselines3 as sb3lib
        model_path = "./submission/model.zip"
        self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self.model.predict(observation=agent_obs, deterministic=True)
            wrapped_act.update({agent_id:action})

        return wrapped_act
