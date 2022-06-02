from typing import Dict

import gym
import numpy as np
from merge.util import plotter3d

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

        if done:
            if obs["events"]["reached_goal"]:
                print(f"ENV: Hooray! Vehicle reached goal.")
            elif obs["events"]["reached_max_episode_steps"]:
                print(f"ENV: Vehicle reached max episode steps.")
            elif (
                obs["events"]["off_road"]
                | obs["events"]["collisions"]
                | obs["events"]["off_route"]
                | obs["events"]["on_shoulder"]
                | obs["events"]["wrong_way"]
            ):
                pass
            else:
                print("Events: ", obs["events"])
                raise Exception("Episode ended for unknown reason.")

            # plot_obs = np.uint8(obs["rgb"])
            # plot_obs_dagm = np.uint8(obs["dagm"])
            # plotter3d(plot_obs, rgb_gray=3,channel_order="last",name="after",pause=0)
            # plotter3d(plot_obs_dagm, rgb_gray=1,channel_order="last",name="after",pause=0)

        return obs, wrapped_reward, done, info

    def _reward(self, obs: Dict[str, gym.Space], env_reward: np.float64) -> np.float64:
        reward = 0

        # Penalty for driving off road
        if obs["events"]["off_road"]:
            reward -= 10
            print(f"ENV: Vehicle went off road.")
            return np.float64(reward)

        # Penalty for driving on road shoulder
        # if obs["events"]["on_shoulder"]:
        #     reward -= 10
        #     print(f"ENV: Vehicle went on road shoulder.")
        #     return np.float64(reward)

        # Penalty for driving on wrong way
        if obs["events"]["wrong_way"]:
            reward -= 10
            print(f"ENV: Vehicle went wrong way.")
            return np.float64(reward)

        # Penalty for colliding
        if obs["events"]["collisions"]:
            reward -= 10
            print(f"ENV: Vehicle collided.")
            return np.float64(reward)

        # Penalty for driving off route
        # if obs["events"]["off_route"]:
        #     reward -= 10
        #     print(f"ENV: Vehicle went off route.")
        #     return np.float64(reward)

        # Reward for reaching goal
        if obs["events"]["reached_goal"]:
            reward += 30
            print(f"ENV: Hooray! Vehicle reached goal. INTRMDT")

        # Reward for distance travelled
        reward += env_reward
        # print("ENV: Added distance reward.")

        return np.float64(reward)
