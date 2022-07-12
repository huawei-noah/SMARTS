from pathlib import Path
from typing import Any, Dict
import numpy as np

# Environment variables (optional)
IMG_METERS = 50  # Observation image area size in meters.
IMG_PIXELS = 256  # Observation image size in pixels.


class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError


def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    from action import Action as DiscreteAction
    from observation import Concatenate, FilterObs
    from reward import Reward

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs
    from smarts.env.wrappers.frame_stack import FrameStack

    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["Continuous"]),
        Reward,
        lambda env: DiscreteAction(env=env, space="Discrete"),
        FilterObs,
        # lambda env: FrameStack(env=env, num_stack=3),
        # lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers


class Policy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        from d3rlpy.algos import CQL


        self.model = CQL.from_json(Path(__file__).absolute().parents[0]/'model/params.json', use_gpu=True)
        self.model.load_model(Path(__file__).absolute().parents[0]/'model/model_274.pt')

        # model_path = Path(__file__).absolute().parents[0] / "best_model.zip"
        # self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            # action, _ = self.model.predict(np.array(agent_obs['rgb'].reshape(3, 256, 256)))
            # action = self.model.predict(np.array([agent_obs['rgb']]))[0]
            agent_x = agent_obs.position[0]
            agent_y = agent_obs.position[1]
            agent_heading = agent_obs.heading
            action = self.model.predict(np.array([agent_x, agent_y, agent_heading]))
            wrapped_act.update({agent_id: action})
        return wrapped_act


class RandomPolicy(BasePolicy):
    """A sample policy with random actions. Note that only the class named `Policy`
    will be tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym

        self._action_space = gym.spaces.Discrete(4)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self._action_space.sample()
            wrapped_act.update({agent_id: action})

        return wrapped_act

policy = Policy()
print(1)