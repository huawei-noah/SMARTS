# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import shutil
import gym
import os
from typing import Any, Dict, Tuple

from smarts.core.sensors import Observation
from smarts.core.utils.file import smarts_log_dir
from smarts.env.wrappers.utils.rendering import (
    make_gif,
    write_image,
    show_notebook_videos,
)

Action = Any
Operation = Any

default_log_dir = smarts_log_dir()


class RecordVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        frequency: int,
        name: str,
        render: bool = True,
        frame_directory: str = default_log_dir,
    ):
        super(RecordVideo, self).__init__(env)
        self._frequency: int = max(frequency, 1)
        self._current_step: int = 0
        self._frame_directory = frame_directory
        self._name = name
        self._render = render
        self._recording_dir = os.path.join(self._frame_directory, self._name)

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        obs, rewards, dones, infos = super().step(action)
        if self._current_step % self._frequency == 0:
            obs_dict = obs
            if isinstance(obs, Observation):
                obs_dict = {"default_agent": obs}
            write_image(obs_dict, self._recording_dir, self._current_step)
        self._current_step += 1

        return (obs, rewards, dones, infos)

    def reset(self) -> Any:
        try:
            os.mkdir(self._recording_dir)
        except:
            pass
        obs = super().reset()
        obs_dict = obs
        if isinstance(obs, Observation):
            obs_dict = {"default_agent": obs}
        write_image(obs_dict, self._recording_dir, self._current_step)
        return obs

    def close(self) -> None:
        if self._render:
            make_gif(self._recording_dir)
            show_notebook_videos(self._recording_dir)
        return super().close()
