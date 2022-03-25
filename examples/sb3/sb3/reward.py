import gym
import numpy as np

from smarts.env.wrappers import format_obs


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        if obs.events["reached_goal"]:
            print(f"***** Vehicle reached goal.")
        if obs.events["reached_max_episode_steps"]:
            print(f"##### Vehicle reached max episode steps.")

        return obs, wrapped_reward, done, info

    def _reward(self, obs: format_obs.StdObs, env_reward: np.float64) -> np.float64:
        reward = 0

        # Penalty for driving off road
        if obs.events["off_road"]:
            reward -= 10
            print(f"----- Vehicle went off road.")
            return np.float64(reward)

        # Penalty for driving on road shoulder
        if obs.events["on_shoulder"]:
            reward -= 10
            print(f"----- Vehicle went on shoulder.")
            return np.float64(reward)

        # Penalty for colliding
        if obs.events["collisions"]:
            reward -= 10
            print(f"----- Vehicle collided.")
            return np.float64(reward)

        # Penalty for driving off route
        if obs.events["off_route"]:
            reward -= 10
            print(f"----- Vehicle went off route.")
            return np.float64(reward)

        # Reward for distance travelled
        reward += env_reward

        return np.float64(reward)
