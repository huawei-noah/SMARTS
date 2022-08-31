import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__file__)

_SCORES_FILENAME = "scores.txt"
_PHASES = ["validation", "track1", "track2"]
_EVALUATION_CONFIG_KEYS = {
    "phase",
    "eval_episodes",
    "seed",
    "scenarios",
    "bubble_env_evaluation_seeds",
}
_DEFAULT_EVALUATION_CONFIG = dict(
    phase="track1",
    eval_episodes=50,
    seed=42,
    scenarios=[
        "1_to_2lane_left_turn_c",
        "1_to_2lane_left_turn_t",
        "3lane_merge_multi_agent",
        "3lane_merge_single_agent",
        "3lane_cruise_multi_agent",
        "3lane_cruise_single_agent",
        "3lane_cut_in",
        "3lane_overtake",
    ],
    bubble_env_evaluation_seeds=[],
)
_SUBMISSION_CONFIG_KEYS = {
    "img_meters",
    "img_pixels",
}
_DEFAULT_SUBMISSION_CONFIG = dict(
    img_meters=64,
    img_pixels=256,
)


def wrap_env(
    env,
    agent_ids: List[str],
    datastore: "DataStore",
    wrappers=[],
):
    """Make environment.

    Args:
        env (gym.Env): The environment to wrap.
        wrappers (List[gym.Wrapper], optional): Sequence of gym environment wrappers.
            Defaults to empty list [].

    Returns:
        gym.Env: Environment wrapped for evaluation.
    """
    # Make a copy of original info.
    env = CopyData(env, agent_ids, datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def evaluate(config):
    base_scenarios = config["scenarios"]
    shared_configs = dict(
        action_space="TargetPose",
        img_meters=int(config["img_meters"]),
        img_pixels=int(config["img_pixels"]),
        sumo_headless=True,
    )
    # Make evaluation environments.
    envs_eval = {}
    for scenario in base_scenarios:
        env = gym.make(
            "smarts.env:multi-scenario-v0", scenario=scenario, **shared_configs
        )
        datastore = DataStore()
        envs_eval[f"{scenario}"] = (
            wrap_env(
                env,
                agent_ids=list(env.agent_specs.keys()),
                datastore=datastore,
                wrappers=submitted_wrappers(),
            ),
            datastore,
            None,
        )

    bonus_eval_seeds = config.get("bubble_env_evaluation_seeds", [])
    for seed in bonus_eval_seeds:
        env = gym.make("bubble_env_contrib:bubble_env-v0", **shared_configs)
        datastore = DataStore()
        envs_eval[f"bubble_env_{seed}"] = (
            wrap_env(
                env,
                agent_ids=list(env.agent_ids),
                datastore=datastore,
                wrappers=submitted_wrappers(),
            ),
            datastore,
            seed,
        )

    # Instantiate submitted policy.
    policy = Policy()

    # Evaluate model for each scenario
    score = Score()
    for index, (env_name, (env, datastore, seed)) in enumerate(envs_eval.items()):
        logger.info(f"\n{index}. Evaluating env {env_name}.\n")
        counts, costs = run(
            env=env,
            datastore=datastore,
            env_name=env_name,
            policy=policy,
            config=config,
            seed=seed,
        )
        score.add(counts, costs)

    rank = score.compute()
    logger.info("\nOverall Rank:\n", rank)
    logger.info("\nFinished evaluating.\n")

    # Close all environments
    for env, _, _ in envs_eval.values():
        env.close()

    return rank


def run(
    env,
    datastore: "DataStore",
    env_name: str,
    policy: "Policy",
    config: Dict[str, Any],
    seed: Optional[int],
):
    # Instantiate metric for score calculation.
    metric = Metric(env_name=env_name, agent_names=datastore.agent_names)

    # Ensure deterministic seeding
    env.seed((seed or 0) + config["seed"])
    eval_episodes = 1 if "naturalistic" in env_name else config["eval_episodes"]
    for _ in range(eval_episodes):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)
            metric.store(infos=datastore.data["infos"], dones=datastore.data["dones"])

    return metric.results()


def to_codalab_scores_string(rank) -> str:
    """Convert the data in scores to a CodaLab-scores-compatible string.

    Note: The score string names must be the same as in the competition.yaml.
    """
    return (
        f"completion: {rank['completion']}\n"
        f"time: {rank['time']}\n"
        f"humanness: {rank['humanness']}\n"
        f"rules: {rank['rules']}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="codalab-evaluation")
    parser.add_argument(
        "--input_dir",
        help=(
            "The path to the directory containing the reference data and user "
            "submission data."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help=(
            "Path to the directory where the submission's scores.txt file will be "
            "written to."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--local",
        help="Flag to set when running evaluate locally. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "--auto_install_pip_deps",
        help="Automatically install dependencies through pip.",
        action="store_true",
    )
    args = parser.parse_args()

    # Get directories.
    from utils import resolve_codalab_dirs

    root_path = str(Path(__file__).absolute().parent)
    submit_dir, evaluation_dir, scores_dir = resolve_codalab_dirs(
        root_path=root_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        local=args.local,
    )

    # Install requirements.
    if args.auto_install_pip_deps:
        from auto_install import install_evaluation_deps

        install_evaluation_deps(
            requirements_dir=Path(root_path), reset_wheelhouse_cache=True
        )

    try:
        import smarts
        import bubble_env_contrib
    except:
        raise ImportError(
            "Missing evaluation dependencies. Please refer to the Setup section of README.md"
            " on how to install the dependencies or use the `--auto_install_pip_deps` flag."
        )

    req_file = os.path.join(submit_dir, "requirements.txt")
    sys.path.insert(0, submit_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

    import gym

    from copy_data import CopyData, DataStore
    from metric import Metric
    from score import Score
    from utils import load_config, merge_config, validate_config, write_output
    from policy import Policy, submitted_wrappers

    # Get config parameters.
    evaluation_config = merge_config(
        self=_DEFAULT_EVALUATION_CONFIG,
        other=load_config(Path(evaluation_dir) / "config.yaml"),
    )
    validate_config(config=evaluation_config, keys=_EVALUATION_CONFIG_KEYS)
    submission_config = merge_config(
        self=_DEFAULT_SUBMISSION_CONFIG,
        other=load_config(Path(submit_dir) / "config.yaml"),
    )
    validate_config(config=submission_config, keys=_SUBMISSION_CONFIG_KEYS)
    config = merge_config(self=evaluation_config, other=submission_config)
    assert config["phase"] in _PHASES, f"Unknown phase config key: {config['phase']}"

    # Run validation, track1, or track2.
    if config["phase"] == "validation":
        rank = evaluate(config)
        rank = dict.fromkeys(rank, 0)
    elif config["phase"] == "track1":
        # Add scenario paths for remote evaluation.
        if not args.local:
            config["scenarios"] = []
            for dirpath, dirnames, filenames in os.walk(evaluation_dir):
                if "scenario.py" in filenames:
                    config["scenarios"].append(dirpath)
        rank = evaluate(config)
    elif config["phase"] == "track2":
        score = Score()
        rank = dict.fromkeys(score.keys, 0)

    text = to_codalab_scores_string(rank)
    output_dir = os.path.join(scores_dir, _SCORES_FILENAME)
    write_output(text=text, output_dir=output_dir)
