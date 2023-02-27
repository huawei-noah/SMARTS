from typing import Any, Dict, Optional

import gymnasium as gym
from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction
from contrib_policy.frame_stack import FrameStack


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

        self.observation_space = self._frame_stack.observation_space
        self._format_action = FormatAction()
        self.action_space = self._format_action.action_space

        print("Policy initialised.")

    def _process(self, obs):
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack(obs)
        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        formatted_action = {
            agent_id: self._format_action.format(agent_action)
            for agent_id, agent_action in action.items()
        }
        obs, reward, terminated, truncated, info = self.env.step(formatted_action)
        obs = {
            agent_id: self._process(agent_obs)
            for agent_id, agent_obs in obs.items()
        }
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""

        self._frame_stack.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process(obs), info
