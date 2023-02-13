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
import gymnasium as gym

from smarts.core.agent_interface import AgentInterface


def checked_bubble_env_v0(
    agent_interface: AgentInterface,
    headless: bool = True,
    seed: int = 42,
    **kwargs,
):
    """Imports and generates the `bubble_env` environment which converts nearby replay
    traffic (non-reactive) into model based traffic (reactive). This uses the NGSIM i80
    dataset as a basis.

    Args:
        traffic_mode (Literal["traffic_A"]): The version of bubble traffic to use.
        action_space (ActionSpaceType): The action space the agent should use.
        img_meters (float): The square side dimensions of the surface the top-down rgb image
            portrays. This affects resolution.
        img_pixels (float): The total number of pixels in the top-down rgb image. This affects
            resolution.
        headless (bool): If the environment should display sumo-gui.
        seed (int): The seed of the environment.


    Returns:
        (gymnasium.Env): The bubble_env environment.

    Raises:
        ImportError: If `bubble_env` is not installed.
    """
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
        agent_interface=agent_interface,
        headless=headless,
        seed=seed,
    )
    return env
