import sys
from pathlib import Path
import gym
import numpy as np
from smarts.core.agent import Agent
from smarts.core.observations import Observation
from typing import Any, Callable, Dict, Tuple

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
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        SaveObs,
        DiscreteAction,
        FilterObs,
        lambda env: FrameStack(env=env, num_stack=3),
        lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers


class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        import stable_baselines3 as sb3lib
        import network

        model_path = Path(__file__).resolve().parents[0] / "saved_model.zip"
        self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Observation):
        """Act function to be implemented by user.

        Args:
            obs (Any): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """

        action, _ = self.model.predict(observation=obs, deterministic=True)
        processed_act = action

        wrapped_act = action_wrapper._discrete(action, self.saved_obs)
        
        return processed_act



def _discrete() -> Tuple[Callable[[Dict[str, int], Dict[str, Any]], Dict[str, np.ndarray]], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    time_delta = 0.1  # Time, in seconds, between steps.
    angle = 5 / 180 * np.pi  # Turning angle in radians
    speed = 30  # Speed in km/h
    dist = (
        speed * 1000 / 3600 * time_delta
    )  # Distance, in meter, travelled in time_delta seconds

    action_map = {
        # key: [magnitude, angle]
        0: [0, 0],  # slow_down
        1: [dist, 0],  # keep_direction
        2: [dist, angle],  # turn_left
        3: [dist, -angle],  # turn_right
    }

    def wrapper(
        action: Dict[str, int], saved_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        wrapped_obs = {}
        for agent_id, agent_action in action.items():
            new_heading = saved_obs[agent_id]["heading"] + action_map[agent_action][1]
            new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

            magnitude = action_map[agent_action][0]
            cur_coord = (
                saved_obs[agent_id]["pos"][0] + 1j * saved_obs[agent_id]["pos"][1]
            )
            # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, numpy_angle = map_angle + π/2
            new_pos = cur_coord + magnitude * np.exp(1j * (new_heading + np.pi / 2))
            x_coord = np.real(new_pos)
            y_coord = np.imag(new_pos)

            wrapped_obs.update(
                {
                    agent_id: np.array(
                        [x_coord, y_coord, new_heading, time_delta], dtype=np.float32
                    )
                }
            )

        return wrapped_obs

    return wrapper, space
