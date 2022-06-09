import gym
import numpy as np
from smarts.core.plan import PositionalGoal

from smarts.core.sensors import Observation


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)

        wrapped_reward = {
            agent_id: self._reward_with_goal(obs[agent_id], agent_reward)
            for agent_id, agent_reward in env_reward.items()
        }

        return obs, wrapped_reward, done, info

    def _reward(self, obs: Observation, env_reward: np.float64) -> float:
        reward = 0

        # Penalty for driving off road
        if obs.events.off_road:
            reward -= 5
            return float(reward)

        # Penalty for colliding
        if len(obs.events.collisions) > 0:
            reward -= 10
            return float(reward)

        # Reward for distance travelled
        reward += env_reward

        return float(reward)

    def _reward_with_goal(self, obs: Observation, env_reward: np.float64) -> float:
        reward = 0

        if obs.events.reached_goal:
            reward += env_reward
            return float(reward)

        # Penalty for driving off road
        if obs.events.off_road:
            reward -= 5
            return float(reward)

        # Penalty for colliding
        if len(obs.events.collisions) > 0:
            reward -= 10
            return float(reward)

        # Reward for distance travelled
        reward += env_reward


        return float(reward)
