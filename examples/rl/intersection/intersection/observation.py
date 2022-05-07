import gym
import numpy as np
from typing import Dict


class Observation(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space["rgb"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[-1],) + old_space.shape[:-1],
            dtype=np.uint8,
        )

    def step(self, action):
        """Steps the environment by one step.

        Args:
            actions (Any): Agent's action.

        Returns:
            Tuple[ np.ndarray, float, bool, Dict[str, Any] ]:
                Observation, reward, done, info, of the agent.
        """
        obs, rewards, dones, infos = self.env.step(action)
        filtered = filter_obs(obs)
        return filtered, rewards, dones, infos

    def reset(self):
        """Resets the environment.

        Returns:
            np.ndarray: Agent's observation after reset.
        """
        obs = self.env.reset()
        filtered = filter_obs(obs)
        return filtered


def filter_obs(obs: Dict[str, gym.Space]) -> np.ndarray:
    rgb = obs["rgb"]

    # Channel first
    rgb = rgb.transpose(2, 0, 1)

    return np.uint8(rgb)