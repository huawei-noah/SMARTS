from typing import Dict

import gym
import numpy as np
from train.util import plotter3d


class FilterObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict(
                    {
                        "rgb": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(agent_obs_space["rgb"].shape[-1],)
                            + agent_obs_space["rgb"].shape[:-1],
                            dtype=np.uint8,
                        ),
                        "goal_distance": gym.spaces.Box(
                            low=-1e10,
                            high=+1e10,
                            shape=(1, 1),
                            dtype=np.float64,
                        ),
                        "goal_heading": gym.spaces.Box(
                            low=-np.pi,
                            high=np.pi,
                            shape=(1, 1),
                            dtype=np.float32,
                        ),
                    }
                )
                for agent_id, agent_obs_space in env.observation_space.spaces.items()
            }
        )

    def observation(
        self, obs: Dict[str, Dict[str, gym.Space]]
    ) -> Dict[str, Dict[str, gym.Space]]:
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            rgb = agent_obs["rgb"]
            rgb = rgb.transpose(2, 0, 1)  # Channel first

            goal_distance = np.array(
                [
                    [
                        np.linalg.norm(
                            agent_obs["mission"]["goal_pos"] - agent_obs["ego"]["pos"]
                        )
                    ]
                ],
                dtype=np.float64,
            )

            ego_heading = (agent_obs["ego"]["heading"] + np.pi) % (2 * np.pi) - np.pi

            goal_x, goal_y = agent_obs["mission"]["goal_pos"][:2]
            goal_angle = np.angle(goal_x + goal_y * 1j) - np.pi / 2
            goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

            goal_heading = goal_angle - ego_heading
            goal_heading = (goal_heading + np.pi) % (2 * np.pi) - np.pi
            goal_heading = np.array([[goal_heading]], dtype=np.float32)

            wrapped_obs.update(
                {
                    agent_id: {
                        "rgb": np.uint8(rgb),
                        "goal_distance": goal_distance,
                        "goal_heading": goal_heading,
                    }
                }
            )

            # print(f"goal_angle {goal_angle*180/np.pi}")
            # print(f"ego_heading {ego_heading*180/np.pi}")
            # print(f"goal_heading {goal_heading}")
            # print(f"goal_distance {goal_distance}")
            # plotter3d(wrapped_obs[agent_id]["rgb"], rgb_gray=3, channel_order="first",name="after",pause=0, save=False)

        return wrapped_obs


class Concatenate(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        _, agent_obs_space = next(iter(env.observation_space.spaces.items()))
        self._num_stack = len(agent_obs_space)
        self.observation_space = gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict(
                    {
                        "rgb": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=agent_obs_space[0]["rgb"].shape[0:-1]
                            + (self._num_stack * agent_obs_space[0]["rgb"].shape[-1],),
                            dtype=np.uint8,
                        ),
                        "goal_error": gym.spaces.Box(
                            low=np.tile(
                                agent_obs_space[0]["goal_error"].low,
                                (self._num_stack, 1),
                            ),
                            high=np.tile(
                                agent_obs_space[0]["goal_error"].high,
                                (self._num_stack, 1),
                            ),
                            shape=(self._num_stack,)
                            + agent_obs_space[0]["goal_error"].shape,
                            dtype=agent_obs_space[0]["goal_error"].dtype,
                        ),
                    }
                )
                for agent_id, agent_obs_space in env.observation_space.spaces.items()
            }
        )

    def observation(self, obs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            rgb, goal_error = zip(
                *[(obs["rgb"], obs["goal_error"]) for obs in agent_obs]
            )
            rgb = np.dstack(rgb)
            goal_error = np.vstack(goal_error)

            wrapped_obs.update(
                {
                    agent_id: {
                        "rgb": np.uint8(rgb),
                        "goal_error": np.float64(goal_error),
                    }
                }
            )
            # plotter3d(wrapped_obs[agent_id]["rgb"], rgb_gray=3,channel_order="last",name="after",pause=0, save=True)

        return wrapped_obs
