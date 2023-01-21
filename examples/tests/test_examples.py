import sys
import tempfile
from pathlib import Path

import pytest

from smarts.core.utils import import_utils

# Necessary to import default_argument_parser in the examples
sys.path.insert(0, str(Path(__file__).parents[1]))

import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)


@pytest.mark.parametrize(
    "example",
    ["egoless", "chase_via_points", "trajectory_tracking", "laner", "hiway_v1"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example):
    if example == "egoless":
        from examples import egoless as current_example
    elif example == "chase_via_points":
        from examples.control import chase_via_points as current_example
    elif example == "trajectory_tracking":
        from examples.control import trajectory_tracking as current_example
    elif example == "laner":
        from examples.control import laner as current_example
    elif example == "hiway_v1":
        from examples.control import hiway_env_v1_lane_follower as current_example
    main = current_example.main
    main(
        scenarios=["scenarios/sumo/loop"],
        headless=True,
        num_episodes=1,
        max_episode_steps=100,
    )


def test_rllib_example():
    from examples.rl.rllib import rllib

    main = rllib.main
    with tempfile.TemporaryDirectory() as result_dir, tempfile.TemporaryDirectory() as model_dir:
        main(
            scenario="scenarios/sumo/loop",
            envision=False,
            time_total_s=20,
            rollout_fragment_length=200,
            train_batch_size=200,
            seed=42,
            num_samples=1,
            num_agents=2,
            num_workers=1,
            resume_training=False,
            result_dir=result_dir,
            checkpoint_num=None,
            save_model_path=model_dir,
        )
