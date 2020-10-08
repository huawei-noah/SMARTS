import gym
import numpy as np

from smarts.contrib.pymarl import PyMARLHiWayEnv


class ListHiWayEnv(PyMARLHiWayEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        super(ListHiWayEnv, self).__init__(config)

    def step(self, agent_actions):
        agent_actions = np.array(agent_actions)
        _, _, infos = super().step(agent_actions)
        n_rewards = infos.pop("rewards_list")
        n_dones = infos.pop("dones_list")
        return (
            np.asarray(self.get_obs()),
            np.asarray(n_rewards),
            np.asarray(n_dones),
            infos,
        )
