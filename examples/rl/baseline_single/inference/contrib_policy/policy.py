import sys
from pathlib import Path

# To import contrib_policy folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from smarts.core.agent import Agent


class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self, config=None, top_down_rgb=None):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        import stable_baselines3 as sb3lib
        from contrib_policy import network
        from contrib_policy.filter_obs import FilterObs
        from contrib_policy.format_action import FormatAction
        from contrib_policy.frame_stack import FrameStack
        from contrib_policy.utils import objdict

        from smarts.core.agent_interface import RGB

        model_path = Path(__file__).resolve().parents[0] / "saved_model.zip"
        self.model = sb3lib.PPO.load(model_path)

        if config == None:
            config = objdict({"num_stack": 3})
        if top_down_rgb == None:
            top_down_rgb = RGB(
                width=112,
                height=112,
                resolution=50 / 112,  # m/pixels
            )

        self._filter_obs = FilterObs(top_down_rgb=top_down_rgb)
        self._frame_stack = FrameStack(
            input_space=self._filter_obs.observation_space,
            num_stack=config.num_stack,
            stack_axis=0,
        )
        self.observation_space = self._frame_stack.observation_space
        self.format_action = FormatAction()
        self._frame_stack.reset()
        print("Policy initialised.")

    def act(self, obs):
        """Act function to be implemented by user."""
        processed_obs = self._process(obs)
        action, _ = self.model.predict(observation=processed_obs, deterministic=True)
        formatted_action = self.format_action.format(action)

        return formatted_action

    def _process(self, obs):
        if obs["steps_completed"] == 1:
            # Reset memory because episode was reset.
            self._frame_stack.reset()
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        return obs
