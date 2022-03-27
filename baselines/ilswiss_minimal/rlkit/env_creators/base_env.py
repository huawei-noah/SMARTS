from abc import ABC, abstractmethod


class BaseEnv(ABC):
    def __init__(self, **configs):
        self.observation_space_n = None
        self.action_space_n = None
        self.agent_ids = None
        self.n_agents = None
        self.configs = configs

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action_n):
        pass
