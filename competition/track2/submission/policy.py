from pathlib import Path
from typing import Any, Dict
import numpy as np
from utility import get_trans_coor


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

    from reward import Reward

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs

    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        Reward,
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

        self.model = CQL.from_json(
            Path(__file__).absolute().parents[0] / "model/params.json", use_gpu=True
        )
        self.model.load_model(Path(__file__).absolute().parents[0] / "model/model.pt")

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self.model.predict(
                np.array([np.moveaxis(agent_obs["rgb"], -1, 0)])
            )[0]
            print(action)
            target_pose = np.array(
                [
                    (action[0] + agent_obs["ego"]["pos"][0]),
                    action[1] + agent_obs["ego"]["pos"][1],
                    action[2] + agent_obs["ego"]["heading"],
                    0.1,
                ]
            )
            wrapped_act.update({agent_id: target_pose})
        return wrapped_act
