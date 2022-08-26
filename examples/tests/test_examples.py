import sys
import tempfile
from pathlib import Path

import psutil
import pytest

from smarts.core.utils import import_utils

# necessary to import default_argument_parser properly in the examples
sys.path.insert(0, str(Path(__file__).parents[1]))

import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)


@pytest.mark.parametrize(
    "example",
    ["egoless", "single_agent", "multi_agent"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example):
    if example == "egoless":
        from examples import egoless as current_example
    if example == "single_agent":
        from examples import single_agent as current_example
    if example == "multi_agent":
        from examples import multi_agent as current_example
    main = current_example.main
    main(
        scenarios=["scenarios/sumo/loop"],
        headless=True,
        num_episodes=1,
        max_episode_steps=100,
    )


def test_ray_multi_instance_example():
    from examples import ray_multi_instance

    main = ray_multi_instance.main
    num_cpus = max(2, min(10, (psutil.cpu_count(logical=False) - 1)))
    main(
        training_scenarios=["scenarios/sumo/loop"],
        evaluation_scenarios=["scenarios/sumo/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
        num_cpus=num_cpus,
    )


def test_rllib_example():
    from examples.rllib import rllib

    main = rllib.main
    with tempfile.TemporaryDirectory() as result_dir, tempfile.TemporaryDirectory() as model_dir:
        main(
            scenario="scenarios/sumo/loop",
            headless=True,
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
