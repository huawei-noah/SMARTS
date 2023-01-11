from typing import Any, Dict, Optional, Sequence, Tuple, Union
import gymnasium as gym
import random

from smarts.zoo.agent_spec import AgentSpec
from gymnasium.spaces import Discrete
from gymnasium.spaces import Tuple as gym_Tuple
from gymnasium.spaces import Dict as gym_Dict


class test_env(gym.Env):
    def __init__(
        self,
        agent_interfaces: Dict[str, AgentSpec],
    ):
        agent_ids = list(agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self.agent_id = agent_ids[0]
        self.action_space = gym_Dict(
            {self.agent_id: gym_Tuple((Discrete(61), Discrete(4, start=-1)))}
        )
        self.observation_space = gym_Dict({self.agent_id: Discrete(4, start=10)})
        self.agent_specs = agent_interfaces

    def step(
        self, agent_actions: Dict
    ) -> Dict:  # (observation, reward, terminated, truncated, info)
        if list(agent_actions.values())[0][0] < 30:
            return (
                {self.agent_id: 10},
                {self.agent_id: 1},
                {self.agent_id: False},
                {self.agent_id: False},
                {self.agent_id: {}},
            )
        else:
            return (
                {self.agent_id: 13},
                {self.agent_id: 2},
                {self.agent_id: True},
                {self.agent_id: True},
                {self.agent_id: {}},
            )

    def reset(self, seed=None, options={}) -> Dict:
        super().reset(seed=seed)
        return {self.agent_id: ({self.agent_id: 10}, {self.agent_id: {}})}

    def render(self):
        return None
