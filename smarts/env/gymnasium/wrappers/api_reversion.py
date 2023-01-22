# MIT License
#
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Any, Dict, SupportsFloat, Tuple

import gym


class Api021Reversion(gym.Wrapper):
    """This wrapper reverts the API of a gym/gymnasium environment to v0.21 of gym."""

    def step(self, action) -> Tuple[Dict, SupportsFloat, bool, Dict[str, Any]]:
        """Converts :meth:`step` return from `obs, reward, term, trunc, info` to `obs, reward, done, info`."""
        obs, reward, terminated, _, info = super().step(action)
        return obs, reward, terminated, info

    def reset(self, **kwargs) -> Dict[str, Tuple[Any]]:
        """Converts :meth:`reset` return from `obs, info` to `obs`."""
        obs, _ = super().reset()
        return obs

    def render(self, mode="human", **kwargs) -> Any:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()
