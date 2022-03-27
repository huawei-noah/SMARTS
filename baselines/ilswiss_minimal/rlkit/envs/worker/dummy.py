import gym
import numpy as np
from typing import Any, List, Callable, Optional, Dict

from rlkit.envs.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env], auto_reset=False) -> None:
        self.env = env_fn()
        super().__init__(env_fn, auto_reset=auto_reset)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.env, key)

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send_action(self, action_n: Dict[str, np.ndarray]) -> None:
        self.result = self.env.step(action_n)
        if all(list(self.result[2].values())) and self.auto_reset:
            self.result[0] = self.env.reset()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()
