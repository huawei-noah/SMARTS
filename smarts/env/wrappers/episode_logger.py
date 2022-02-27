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
import gym
from typing import Any, Dict, Tuple
from smarts.core.utils.episodes import EpisodeLog, EpisodeLogs

Action = Any
Operation = Any


class EpisodeLogger(gym.Wrapper):
    def __init__(self, env: gym.Env, col_width: int = 18):
        super(EpisodeLogger, self).__init__(env)
        self._first_step = False
        self._current_episode = None
        self._done = False
        self._log_iter = self._episode_logs(col_width)

    def step(self, action: Action) -> Tuple[Operation, float, bool, Dict[str, Any]]:
        step_vals = super().step(action)
        self._current_episode.record_step(*step_vals)
        return step_vals

    def reset(self) -> Any:
        obs = super().reset()
        self._current_episode: EpisodeLog = next(self._log_iter)
        self._current_episode.record_scenario(self.scenario_log)
        return obs

    def close(self):
        self._done = True
        try:
            next(self._log_iter)
        except:
            pass
        return super().close()

    def _episode_logs(self, col_width) -> EpisodeLog:
        with EpisodeLogs(col_width) as episode_logs:
            while not self._done:
                yield episode_logs.reset()
