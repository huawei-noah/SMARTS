from typing import Any, Callable, Dict

import gym
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        env: gym.Env,
        video_length: int,
        record_video_trigger: Callable[[int], bool],
        name_prefix: str = "rl-video",
    ):
        """
        Records a video of an agent's trajectory traversing `env` and logs it to TensorBoard

        :param env: A gym environment from which the trajectory is recorded
        :param video_length: Length of recorded videos
        :param record_video_trigger: Function that defines when to start recording.
            The function takes the current number of step, and returns whether we should start recording or not.
        :param name_prefix: Prefix to the video name
        """

        super().__init__()
        self._env = env
        self._video_length = video_length

    def _on_step(self) -> bool:
        if self.n_calls < self._video_length:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            # evaluate_policy(
            #     self.model,
            #     self._eval_env,
            #     callback=grab_screens,
            #     n_eval_episodes=self._n_eval_episodes,
            #     deterministic=self._deterministic,
            # )

            # self.locals
            # self.globals
            # import sys
            # sys.exit(3)

            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
