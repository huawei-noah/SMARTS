import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Any, Dict
from smarts.env.wrappers.format_action import FormatAction
from smarts.env.wrappers.format_obs import FormatObs
from smarts.core.controllers import ActionSpaceType

# To import utility folder
sys.path.insert(0, str(Path(__file__).parents[0]))

from utility import (
    get_goal_layer,
    global_target_pose,
)


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

    # Insert wrappers here, if any.
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
    ]

    return wrappers


class Policy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        # Load saved model and instantiate any needed objects.
        from d3rlpy.algos import CQL

        policy_name = [
            policy_name
            for policy_name in os.listdir(Path(__file__).absolute().parents[0])
            if policy_name.endswith("pt")
        ][0]
        self.policy = torch.jit.load(Path(__file__).absolute().parents[0] / policy_name)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """

        # Use saved model to predict multi-agent action output given multi-agent SMARTS observation input.
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
            final_obs = np.array(final_obs)
            final_obs = torch.tensor(final_obs, dtype=torch.float32)

            action = self.policy(final_obs)[0]
            target_pose = global_target_pose(action.cpu().numpy(), agent_obs)
            wrapped_act.update({agent_id: target_pose})

        return wrapped_act
