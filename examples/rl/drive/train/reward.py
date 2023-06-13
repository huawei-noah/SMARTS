from typing import Tuple

import gymnasium as gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, crop: Tuple[int, int, int, int]):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self._total_dist = {}
        self._crop = crop

    def reset(self, *, seed=None, options=None):
        self._total_dist = {}
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_obs in obs.items():
            # Accumulate total distance travelled
            self._total_dist[agent_id] = (
                self._total_dist.get(agent_id, 0) + agent_obs["distance_travelled"]
            )

            # If agent is done
            if terminated[agent_id] == True:
                if agent_obs["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                elif agent_obs["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    agent_obs["events"]["collisions"]
                    | agent_obs["events"]["off_road"]
                    | agent_obs["events"]["off_route"]
                    | agent_obs["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", agent_obs["events"])
                    raise Exception("Episode ended for unknown reason.")

                print(
                    f"{agent_id}: Steps = {agent_obs['steps_completed']} "
                    f"{agent_id}: Dist = {self._total_dist[agent_id]:.2f}"
                )

        return obs, wrapped_reward, terminated, truncated, info

    def _reward(self, obs, env_reward):
        reward = {agent_id: np.float64(0) for agent_id in obs.keys()}

        for agent_id, agent_obs in obs.items():
            # Penalty for colliding
            if agent_obs["events"]["collisions"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Collided.")
                continue

            # Penalty for driving off road
            if agent_obs["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                continue

            # Penalty for driving off route
            if obs[agent_id]["events"]["off_route"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off route.")
                continue

            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went wrong way.")
                continue

            # Reward for reaching goal
            if agent_obs["events"]["reached_goal"]:
                reward[agent_id] += np.float64(30)

            # Reward for distance travelled in one step
            reward[agent_id] += np.float64(env_reward[agent_id])

        return reward
