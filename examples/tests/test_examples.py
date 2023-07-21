import sys
import tempfile
from pathlib import Path
from importlib import import_module
from typing import Literal
from hydra import initialize_config_dir, compose

import pytest

from smarts.core.utils import import_utils

# Necessary to import default_argument_parser in the examples
sys.path.insert(0, str(Path(__file__).parents[1]))

import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)


@pytest.mark.parametrize(
    "example",
    ["1_egoless", "2_single_agent", "3_multi_agent", "4_environment_config", "5_agent_zoo", "6_experiment_base"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example: Literal['1_egoless', '2_single_agent', '3_multi_agent', '4_environment_config', '5_agent_zoo', '6_experiment_base']):
    current_example = import_module(example, "examples")
    main = current_example.main

    if example != "6_experiment_base":
        main(
            scenarios=["scenarios/sumo/loop"],
            headless=True,
            num_episodes=1,
            max_episode_steps=100,
        )
    else:
        example_path = Path(current_example.__file__).parent
        with initialize_config_dir(version_base=None, config_dir=str(example_path.absolute()/"configs"/example)):
            cfg = compose(config_name="experiment_default")
            main(cfg)


def test_rllib_pg_example():
    from examples.rl.rllib import pg_example

    main = pg_example.main
    with tempfile.TemporaryDirectory() as result_dir:
        main(
            scenarios=["./scenarios/sumo/loop"],
            envision=False,
            time_total_s=20,
            rollout_fragment_length=200,
            train_batch_size=200,
            seed=42,
            num_agents=2,
            num_workers=1,
            resume_training=False,
            result_dir=result_dir,
            checkpoint_num=None,
            checkpoint_freq=1,
            log_level="WARN",
        )


def test_rllib_tune_pg_example():
    from examples.rl.rllib import pg_pbt_example

    main = pg_pbt_example.main
    with tempfile.TemporaryDirectory() as result_dir, tempfile.TemporaryDirectory() as model_dir:
        main(
            scenarios=["./scenarios/sumo/loop"],
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
            checkpoint_freq=1,
            save_model_path=model_dir,
            log_level="WARN",
        )
