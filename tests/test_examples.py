import tempfile
import importlib
import pytest


@pytest.mark.parametrize(
    "example",
    ["egoless", "single_agent", "multi_agent"],
    # TODO: "ego_open_agent" and "human_in_the_loop" are causing aborts, fix later
)
def test_examples(example):
    main = importlib.import_module(f"examples.{example}").main
    main(
        scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
        max_episode_steps=100,
    )


def test_multi_instance_example():
    main = importlib.import_module("examples.multi_instance").main
    main(
        training_scenarios=["scenarios/loop"],
        evaluation_scenarios=["scenarios/loop"],
        sim_name=None,
        headless=True,
        num_episodes=1,
        seed=42,
    )


def test_rllib_example():
    main = importlib.import_module("examples.rllib").main
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
