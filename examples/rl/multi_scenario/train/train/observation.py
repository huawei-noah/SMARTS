from typing import Dict

import gym
import numpy as np
from train.util import plotter3d


class FilterObs(gym.ObservationWrapper):
    """Filter only the selected observation parameters."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
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
                            dtype=np.float32,
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
            # Channel first rgb
            rgb = agent_obs["rgb"]
            rgb = rgb.transpose(2, 0, 1)

            # Distance between ego and goal.
            goal_distance = np.array(
                [
                    [
                        np.linalg.norm(
                            agent_obs["mission"]["goal_pos"] - agent_obs["ego"]["pos"]
                        )
                    ]
                ],
                dtype=np.float32,
            )

            # Ego's heading with respect to the map's axes.
            # Note: All angles returned by smarts is with respect to the map's axes.
            #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
            ego_heading = (agent_obs["ego"]["heading"] + np.pi) % (2 * np.pi) - np.pi
            ego_pos = agent_obs["ego"]["position"]

            # Goal's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            goal_pos = agent_obs["mission"]["goal_pos"]
            rel_pos = goal_pos - ego_pos
            goal_angle = np.angle(rel_pos[0] + 1j*rel_pos[1]) - np.pi / 2
            goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

            # Goal heading is the angle correction required by ego agent to face the goal.
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
            # print(f"ego_angle {ego_angle*180/np.pi}")
            # print(f"goal_heading {goal_heading}")
            # print(f"goal_distance {goal_distance}")
            # plotter3d(wrapped_obs[agent_id]["rgb"], rgb_gray=3, channel_order="first",name="after",pause=0, save=True)

        return wrapped_obs


class Concatenate(gym.ObservationWrapper):
    """Concatenates data from stacked dictionaries. Only works with nested gym.spaces.Box .
    Dimension to stack over is determined by `channels_order`.
    """

    def __init__(self, env: gym.Env, channels_order: str = "first"):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
            channels_order (str): A string, either "first" or "last", specifying
                the dimension over which to stack each observation.
        """
        super().__init__(env)

        self._repeat_axis = {
            "first": 0,
            "last": -1,
        }.get(channels_order)

        for agent_name, agent_space in env.observation_space.spaces.items():
            for subspaces in agent_space:
                for key, space in subspaces.spaces.items():
                    assert isinstance(space, gym.spaces.Box), (
                        f"Concatenate only works with nested gym.spaces.Box. "
                        f"Got agent {agent_name} with key {key} and space {space}."
                    )

        _, agent_space = next(iter(env.observation_space.spaces.items()))
        self._num_stack = len(agent_space)
        self._keys = agent_space[0].spaces.keys()

        obs_space = {}
        for agent_name, agent_space in env.observation_space.spaces.items():
            subspaces = {}
            for key, space in agent_space[0].spaces.items():
                low = np.repeat(space.low, self._num_stack, axis=self._repeat_axis)
                high = np.repeat(space.high, self._num_stack, axis=self._repeat_axis)
                subspaces[key] = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
            obs_space.update({agent_name: gym.spaces.Dict(subspaces)})
        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        """Adapts the wrapped environment's observation.

        Note: Users should not directly call this method.
        """

        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            stacked_obs = {}
            for key in self._keys:
                val = [obs[key] for obs in agent_obs]
                stacked_obs[key] = np.concatenate(val, axis=self._repeat_axis)
            wrapped_obs.update({agent_id: stacked_obs})

            # plotter3d(wrapped_obs[agent_id]["rgb"], rgb_gray=3,channel_order="first",name="after",pause=0, save=True)

        return wrapped_obs
