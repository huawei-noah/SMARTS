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
from textwrap import dedent
from typing import Optional

import gymnasium as gym


def checked_bubble_env_v0(
    traffic_mode="traffic_A",
    action_space="Direct",
    img_meters: int = 64,
    img_pixels: int = 256,
    headless: bool = True,
    seed: int = 42,
    **kwargs,
):

    try:
        import bubble_env_contrib
    except ImportError as err:
        raise ImportError(
            """
            Bubble env is not installed.
            
            If bubble_env is not installed, please install the bubble_env repository:

            ```bash
            # set $REPOS to wherever you wish to store the repository.
            git lfs clone https://bitbucket.org/malban/bubble_env.git $REPOS/bubble_env
            # read $REPOS/README.md and follow those instructions
            cd -
            pip install $REPOS/bubble_env
            ```
            """,
            "Install bubble_env",
        ) from err

    env = gym.make(
        "bubble_env_contrib:bubble_env-v1",
        action_space=action_space,
        img_meters=img_meters,
        img_pixels=img_pixels,
        headless=headless,
        seed=seed,
        traffic_mode=traffic_mode,
    )
    return env
