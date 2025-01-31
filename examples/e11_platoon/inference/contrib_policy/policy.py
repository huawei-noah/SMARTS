import sys
from pathlib import Path

# To import contrib_policy folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from smarts.core.agent import Agent


class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self, num_stack, top_down_rgb, crop, action_space_type):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        from contrib_policy import network
        from contrib_policy.filter_obs import FilterObs
        from contrib_policy.format_action import FormatAction
        from contrib_policy.frame_stack import FrameStack
        from contrib_policy.make_dict import MakeDict

        self._model = self._get_model()

        self._filter_obs = FilterObs(top_down_rgb=top_down_rgb, crop=crop)
        self._frame_stack = FrameStack(
            input_space=self._filter_obs.observation_space,
            num_stack=num_stack,
            stack_axis=0,
        )
        self._frame_stack.reset()
        self._make_dict = MakeDict(input_space=self._frame_stack.observation_space)

        self.observation_space = self._make_dict.observation_space

        self._format_action = FormatAction(action_space_type=action_space_type)
        self.action_space = self._format_action.action_space

    def act(self, obs):
        """Mandatory act function to be implemented by user."""
        processed_obs = self._process(obs)
        action, _ = self._model.predict(observation=processed_obs, deterministic=True)
        formatted_action = self._format_action.format(
            action=int(action), prev_heading=obs["ego_vehicle_state"]["heading"]
        )
        return formatted_action

    def _process(self, obs):
        if obs["steps_completed"] == 1:
            # Reset memory because episode was reset.
            self._frame_stack.reset()
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        obs = self._make_dict.make(obs)
        return obs

    def _get_model(self):
        import stable_baselines3 as sb3lib

        return sb3lib.PPO.load(path=Path(__file__).resolve().parents[0] / "saved_model")
