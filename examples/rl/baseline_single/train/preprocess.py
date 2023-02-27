from typing import Any, Dict, Optional

import gymnasium as gym
from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction
from contrib_policy.frame_stack import FrameStack


class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, config, top_down_rgb):
        super().__init__(env)

        self._agent_ids = env.agent_ids
        self._filter_obs = {
            agent_id: FilterObs(top_down_rgb=top_down_rgb)
            for agent_id in self._agent_ids
        }
        self._frame_stack = {
            agent_id: FrameStack(
                input_space=self._filter_obs[agent_id].observation_space,
                num_stack=config.num_stack,
                stack_axis=0,
            )
            for agent_id in self._agent_ids
        }
        for agent_id in self._agent_ids:
            self._frame_stack[agent_id].reset()

        self.observation_space = gym.spaces.Dict({
            agent_id: self._frame_stack[agent_id].observation_space
            for agent_id in self._agent_ids
        })
        self._format_action = {
            agent_id: FormatAction()
            for agent_id in self._agent_ids
        }
        self.action_space = gym.spaces.Dict({
            agent_id: self._format_action[agent_id].action_space
            for agent_id in self._agent_ids
        })

        print("Policy initialised.")

    def _process(self, agent_id, obs):
        obs = self._filter_obs[agent_id].filter(obs)
        obs = self._frame_stack[agent_id].stack(obs)
        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        formatted_action = {
            agent_id: self._format_action[agent_id].format(agent_action)
            for agent_id, agent_action in action.items()
        }
        obs, reward, terminated, truncated, info = self.env.step(formatted_action)
        obs = {
            agent_id: self._process(agent_id, agent_obs)
            for agent_id, agent_obs in obs.items()
        }
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""

        for agent_id in self._agent_ids:
            self._frame_stack[agent_id].reset()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = {
            agent_id: self._process(agent_id, agent_obs)
            for agent_id, agent_obs in obs.items()
        }
        return obs, info
