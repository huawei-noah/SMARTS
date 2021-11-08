import tempfile
import importlib
import pytest
import sys

import importlib.util

from pathlib import Path

examples_dir = str(Path(__file__).parents[1]) # Get one directory up
if examples_dir not in sys.path:
    sys.path.append(examples_dir)

examples_name = "smarts_examples"

spec = importlib.util.spec_from_file_location(examples_name, f"{examples_dir}/__init__.py")
module = importlib.util.module_from_spec(spec)
sys.modules[examples_name] = module
spec.loader.exec_module(module)

@pytest.mark.parametrize(
    "example",
    ["egoless", "single_agent", "multi_agent"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example):
    if example == "egoless":
        from smarts_examples import egoless as current_example
    if example == "single_agent":
        from smarts_examples import single_agent as current_example
    if example == "multi_agent":
        from smarts_examples import multi_agent as current_example
    main = current_example.main
    main(
        scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
        max_episode_steps=100,
    )


def test_multi_instance_example():
    from smarts_examples import multi_instance
    main = multi_instance.main
    main(
        training_scenarios=["scenarios/loop"],
        evaluation_scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
    )


def test_rllib_example():
    from smarts_examples import rllib
    main = rllib.main
    with tempfile.TemporaryDirectory() as result_dir, tempfile.TemporaryDirectory() as model_dir:
        main(
            scenario="scenarios/loop",
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
