import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Callable


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env], auto_reset=False) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        self.action_space_n = getattr(self, "action_space_n")
        self.auto_reset = auto_reset

    @abstractmethod
    def __getattr__(self, key: str) -> Any:
        pass

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        pass

    def get_result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.result

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform one timestep of the environment's dynamic.
        "send_action" and "get_result" are coupled in sync simulation, so
        typically users only call "step" function. But they can be called
        separately in async simulation, i.e. someone calls "send_action" first,
        and calls "get_result" later.
        """
        self.send_action(action)
        return self.get_result()

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        result = dict()
        for agent_id, action_space in self.action_space_n.items():
            action_space.seed(seed)
            result[agent_id] = action_space.seed(seed)
        return result

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()
