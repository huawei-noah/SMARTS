from typing import Dict

import gym
import numpy as np
from merge.util import plotter3d


class FilterObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            agent_id: gym.spaces.Dict({
                "rgb": agent_obs_space["rgb"],
                "goal_pos": agent_obs_space["mission"]["goal_pos"],
            })
            for agent_id, agent_obs_space in env.observation_space.spaces.items()
        })


    def observation(self, obs: Dict[str, Dict[str, gym.Space]]) -> Dict[str, Dict[str, gym.Space]]:
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {} 
        for agent_id, agent_obs in obs.items():
            rgb = agent_obs["rgb"]
            goal_error = agent_obs["mission"]["goal_pos"] - agent_obs["ego"]["pos"]
            wrapped_obs.update(
                {
                    agent_id: {
                        "rgb": np.uint8(rgb), 
                        "goal_error": np.float64(goal_error),
                    }
                }
            )
     
            # plotter3d(wrapped_obs[agent_id]["rgb"], rgb_gray=3,channel_order="last",name="after",pause=-1, save=True)

        return wrapped_obs


class Concatenate(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=old_space.shape[1:-1] + (old_space.shape[0] * old_space.shape[-1],),
            dtype=np.uint8,
        )
        # old_rgb_space = env.observation_space.spaces[]
        # self.observation_space = gym.spaces.Dict({
        #     agent_id: gym.spaces.Dict({
        #         "rgb": gym.spaces.Box(
        #             low=0,
        #             high=255,
        #             shape=old_rgb_space.shape[1:-1] + (old_rgb_space.shape[0] * old_rgb_space.shape[-1],),
        #             dtype=np.uint8,
        #         ),
        #         "goal_pos": agent_obs_space["mission"]["goal_pos"],
        #     })
        #     for agent_id, agent_obs_space in env.observation_space.spaces.items()
        # })

    def observation(self, obs):
        # print("Before:",obs.shape)
        # plotter3d(obs,rgb_gray=3,channel_order="last",name="before")
        assert len(obs.shape) == 4
        obs = np.concatenate(obs[:], axis=-1)
        # print("After:", obs.shape)
        # plotter3d(obs, rgb_gray=3,channel_order="last",name="after",pause=1)
        return obs
