import gym

from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction
from contrib_policy.frame_stack import FrameStack
from contrib_policy.make_dict import MakeDict

class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, config, top_down_rgb):
        super().__init__(env)

        self._filter_obs = FilterObs(top_down_rgb=top_down_rgb)
        self._frame_stack = FrameStack(
                input_space=self._filter_obs.observation_space,
                num_stack=config.num_stack,
                stack_axis=0,
            )
        self._frame_stack.reset()
        self._make_dict = MakeDict(input_space=self._frame_stack.observation_space)

        self.observation_space = self._make_dict.observation_space

        self._format_action = FormatAction()
        self.action_space = self._format_action.action_space
        print("Policy initialised.")

    def _process(self, obs):
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        obs = self._make_dict.make(obs)
        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        formatted_action = self._format_action.format(action)
        obs, reward, done, info = self.env.step(formatted_action)
        obs = self._process(obs)
        return obs, reward, done, info

    def reset(self):
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""

        self._frame_stack.reset()
        obs = self.env.reset()
        obs = self._process(obs)
        return obs
