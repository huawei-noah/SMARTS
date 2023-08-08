import gymnasium as gym
from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction
from contrib_policy.frame_stack import FrameStack
from contrib_policy.make_dict import MakeDict

from smarts.zoo.agent_spec import AgentSpec


class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_spec: AgentSpec):
        super().__init__(env)

        self._filter_obs = FilterObs(
            top_down_rgb=agent_spec.interface.top_down_rgb,
            crop=agent_spec.agent_params["crop"],
        )
        self._frame_stack = FrameStack(
            input_space=self._filter_obs.observation_space,
            num_stack=agent_spec.agent_params["num_stack"],
            stack_axis=0,
        )
        self._frame_stack.reset()
        self._make_dict = MakeDict(input_space=self._frame_stack.observation_space)

        self.observation_space = self._make_dict.observation_space

        self._format_action = FormatAction(
            action_space_type=agent_spec.interface.action
        )
        self.action_space = self._format_action.action_space

    def _process(self, obs):
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        obs = self._make_dict.make(obs)

        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        formatted_action = self._format_action.format(action)
        obs, reward, terminated, truncated, info = self.env.step(formatted_action)
        obs = self._process(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""

        self._frame_stack.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._process(obs)
        return obs, info
