import gym
import numpy as np
import tensorflow as tf

from collections import deque
from smarts.env.wrappers import format_obs


class Observation(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=self._num_stack)

        self.observation_space = env.observation_space.spaces["rgb"]
        print(self.observation_space)
        print("----------------------------------------")

    def _stack_obs(self, obs: format_obs.StdObs):
        self._frames.appendleft(obs)
        stacked_obs = np.vstack(self._frames)
        return stacked_obs

    def step(self, action):
        """Steps the environment by one step.

        Args:
            actions (Any): Agent's action.

        Returns:
            Tuple[ np.ndarray, float, bool, Dict[str, Any] ]: 
                Observation, reward, done, info, of the agent.
        """
        obs, rewards, dones, infos = self.env.step(action)

        return self._stack_obs(obs), rewards, dones, infos

    def reset(self):
        """Resets the environment.

        Returns:
            np.ndarray: Agent's observation after reset.
        """
        obs = self.env.reset()
        for _ in range(self._num_stack - 1):
            self._frames.appendleft(obs)

        return self._stack_obs(obs)
