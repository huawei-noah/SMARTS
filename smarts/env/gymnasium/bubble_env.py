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
