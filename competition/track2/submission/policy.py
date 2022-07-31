from typing import Any, Dict
import numpy as np
from utility import get_goal_layer, global_target_pose


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

    wrappers = [
        # Insert wrappers here, if any.
    ]

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
        self.model.load_model(
            Path(__file__).absolute().parents[0] / "model/model_100.pt"
        )

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """

        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            bev_obs = np.moveaxis(agent_obs["rgb"], -1, 0)

            goal_x = agent_obs["mission"]["goal_pos"][0]
            goal_y = agent_obs["mission"]["goal_pos"][1]
            current_x = agent_obs["ego"]["pos"][0]
            current_y = agent_obs["ego"]["pos"][1]
            current_heading = agent_obs["ego"]["heading"]
            goal_obs = get_goal_layer(
                goal_x, goal_y, current_x, current_y, current_heading
            )

            final_obs = list()
            final_obs.append(np.concatenate((bev_obs, goal_obs), axis=0))
            final_obs = np.array(final_obs, dtype=np.uint8)
            action = self.model.predict(final_obs)[0]
            print(action)
            target_pose = global_target_pose(action, agent_obs)
            wrapped_act.update({agent_id: target_pose})
        return wrapped_act
