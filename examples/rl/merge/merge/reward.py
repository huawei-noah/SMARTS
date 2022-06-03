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

        for agent_id, done in done.items():
            if done:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Vehicle reached goal.")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Vehicle reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["off_road"]
                    | obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_route"]
                    | obs[agent_id]["events"]["on_shoulder"]
                    | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

                # plot_obs = np.uint8(obs["rgb"])
                # plot_obs_dagm = np.uint8(obs["dagm"])
                # plotter3d(plot_obs, rgb_gray=3,channel_order="last",name="after",pause=0)
                # plotter3d(plot_obs_dagm, rgb_gray=1,channel_order="last",name="after",pause=0)

        return obs, wrapped_reward, done, info

    def _reward(self, obs: Dict[str, Dict[str, gym.Space]], env_reward: Dict[str, np.float64]) -> Dict[str, np.float64]:
        reward = {agent_id:np.float64(0) for agent_id in env_reward.keys()}

        for agent_id, agent_reward in env_reward.items():
            # Penalty for driving off road
            if obs[agent_id]["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Vehicle went off road.")
                break

            # Penalty for driving on road shoulder
            # if obs[agent_id]["events"]["on_shoulder"]:
            #     reward[agent_id] -= np.float64(10)
            #     print(f"{agent_id}: Vehicle went on road shoulder.")
            #     break

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Vehicle went wrong way.")
                break

            # Penalty for colliding
            if obs[agent_id]["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Vehicle collided.")
                break

            # Penalty for driving off route
            # if obs[agent_id]["events"]["off_route"]:
            #     reward[agent_id] -= np.float64(10)
            #     print(f"{agent_id}: Vehicle went off route.")
            #     break

            # Reward for reaching goal
            if obs[agent_id]["events"]["reached_goal"]:
                reward[agent_id] += np.float64(30)
                print(f"{agent_id}: Hooray! Vehicle reached goal. INTRMDT")
                
            # Reward for distance travelled
            reward[agent_id] += np.float64(agent_reward)


        return reward