import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

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
    eval_episodes=1,
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
    bubble_env_evaluation_seeds=[6],
)
_SUBMISSION_CONFIG_KEYS = {
    "img_meters",
    "img_pixels",
}
_DEFAULT_SUBMISSION_CONFIG = dict(
    img_meters=64,
    img_pixels=256,
)


def _make_env(
    env_type: str,
    scenario: Optional[str],
    shared_configs: Dict[str, Any],
    seed: Optional[int],
    wrapper_ctors: Callable[[], Sequence["gym.Wrapper"]],
) -> Tuple["gym.Env", "DataStore"]:
    """Build env.

    Args:
        env_type (str): Env type.
        scenario (Optional[str]): Scenario name or path to scenario folder.
        shared_configs (Dict[str, Any]): Env configs.
        seed (Optional[int]): Env seed.
        wrapper_ctors (Callable[[],Sequence[gym.Wrapper]]): Sequence of gym environment wrappers.

    Raises:
        ValueError: If unknown env type is supplied.

    Returns:
        Tuple[gym.Env, "DataStore"]: Wrapped environment and the datastore storing the observations.
    """

    # Make env.
    if env_type == "smarts.env:multi-scenario-v0":
        env = gym.make(env_type, scenario=scenario, **shared_configs)
    elif env_type == "bubble_env_contrib:bubble_env-v0":
        env = gym.make(env_type, **shared_configs)
    else:
        raise ValueError("Unknown env type.")

    # Make datastore.
    datastore = DataStore()
    # Make a copy of original info.
    env = CopyData(env, list(env.agent_specs.keys()), datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment.
    wrappers = wrapper_ctors()
    for wrapper in wrappers:
        env = wrapper(env)

    # Set seed.
    env.seed(seed)

    return env, datastore


def evaluate(config):
    shared_configs = dict(
        action_space="TargetPose",
        img_meters=int(config["img_meters"]),
        img_pixels=int(config["img_pixels"]),
        sumo_headless=True,
    )
    # Make environment constructors.
    env_ctors = {}
    for scenario in config["scenarios"]:
        env_ctors[f"{scenario}"] = partial(
            _make_env,
            env_type="smarts.env:multi-scenario-v0",
            scenario=scenario,
            shared_configs=shared_configs,
            seed=config["seed"],
            wrapper_ctors=submitted_wrappers,
        )
    for seed in config["bubble_env_evaluation_seeds"]:
        env_ctors[f"bubble_env_{seed}"] = partial(
            _make_env,
            env_type="bubble_env_contrib:bubble_env-v0",
            scenario=None,
            shared_configs=shared_configs,
            seed=seed + config["seed"],
            wrapper_ctors=submitted_wrappers,
        )

    # Instantiate submitted policy.
    score = Score()

    # Multiprocessed evaluation.
    with ProcessPoolExecutor(max_workers=3) as pool:
        # Submit all tasks and get future objects
        futures = [
            pool.submit(
                _worker, cloudpickle.dumps([env_name, env_ctor, Policy, config])
            )
            for env_name, env_ctor in env_ctors.items()
        ]
        # Process results from tasks in order of task completion
        for future in as_completed(futures):
            # Get the result
            counts, costs = future.result()
            score.add(counts, costs)

    # for index, (env_name, env_ctor) in enumerate(env_ctors.items()):
    #     logger.info(f"\n{index}. Evaluating env {env_name}.\n")
    #     counts, costs = run(
    #         env_name=env_name,
    #         env_ctor=env_ctor,
    #         policy_ctor=Policy,
    #         config=config,
    #     )
    #     score.add(counts, costs)

    rank = score.compute()
    logger.info(f"\nOverall Rank: {rank}\n")
    logger.info("\nFinished evaluating.\n")

    return rank


def _worker(input: bytes) -> Tuple[Counts, Costs]:
    """Compute metrics of a given env.

    Args:
        input (bytes): cloudpickle of [env_name: str, env_ctor: Callable[[], gym.Env],
            policy_ctor: Callable[[], Policy], config: Dict[str, Any]]

    Returns:
        Tuple[Counts, Costs]: Count and cost metrics.
    """

    env_name, env_ctor, policy_ctor, config = cloudpickle.loads(input)
    env: gym.Env
    datastore: DataStore
    env, datastore = env_ctor()
    policy = policy_ctor()

    # Instantiate metric for score calculation.
    metric = Metric(env_name=env_name, agent_names=datastore.agent_names)

    eval_episodes = 1 if "naturalistic" in env_name else config["eval_episodes"]
    for _ in range(eval_episodes):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)
            metric.store(infos=datastore.data["infos"], dones=datastore.data["dones"])

    env.close()

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
            "Dependencies are missing. Run this evaluation script with "
            "`--auto_install_pip_deps` flag to install dependencies."
        )

    req_file = os.path.join(submit_dir, "requirements.txt")
    sys.path.insert(0, submit_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

    import gym
    import cloudpickle
    from copy_data import CopyData, DataStore
    from costs import Costs
    from counts import Counts
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
