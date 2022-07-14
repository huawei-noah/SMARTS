import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

_SCORES_FILENAME = "scores.txt"
_EVALUATION_CONFIG_KEYS = {
    "img_meters",
    "img_pixels",
    "validate",
    "evaluate",
    "eval_episodes",
    "seed",
    "scenarios",
}
_DEFAULT_EVALUATION_CONFIG = dict(
    validate=True,
    evaluate=True,
    eval_episodes=2,
    seed=42,
    scenarios=[
        "1_to_2lane_left_turn_c",
        # "1_to_2lane_left_turn_t",
        # "3lane_merge_multi_agent",
        # "3lane_merge_single_agent",
        # "3lane_cruise_multi_agent",
        # "3lane_cruise_single_agent",
        # "3lane_cut_in",
        # "3lane_overtake",
    ],
)
_SUBMISSION_CONFIG_KEYS = {
    "img_meters",
    "img_pixels",
}
_DEFAULT_SUBMISSION_CONFIG = dict(img_meters=50, img_pixels=112,)


def make_env(
    config: Dict[str, Any], scenario: str, datastore: "DataStore", wrappers=[],
):
    """Make environment.

    Args:
        config (Dict[str, Any]): A dictionary of config parameters.
        scenario (str): Scenario
        wrappers (List[gym.Wrapper], optional): Sequence of gym environment wrappers.
            Defaults to empty list [].

    Returns:
        gym.Env: Environment corresponding to the `scenario`.
    """

    # Create environment
    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario=scenario,
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        action_space="Continuous",
        sumo_headless=True,
    )

    # Make a copy of original info.
    env = CopyData(env, datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def evaluate(evaluation_config, submission_config):
    config = resolve_config(submission_config, evaluation_config)
    scenarios = config["scenarios"]

    # Make evaluation environments.
    envs_eval = {}
    for scen in scenarios:
        datastore = DataStore()
        envs_eval[f"{scen}"] = (
            make_env(
                config=config,
                scenario=scen,
                datastore=datastore,
                wrappers=submitted_wrappers(),
            ),
            datastore,
        )

    # Instantiate submitted policy.
    policy = Policy()

    # Evaluate model for each scenario
    score = Score()
    for index, (env_name, (env, datastore)) in enumerate(envs_eval.items()):
        logger.info(f"\n{index}. Evaluating env {env_name}.\n")
        counts, costs = run(
            env=env,
            datastore=datastore,
            env_name=env_name,
            policy=policy,
            config=config,
        )
        score.add(counts, costs)

    rank = score.compute()
    logger.info("\nOverall Rank:\n", rank)
    logger.info("\nFinished evaluating.\n")

    # Close all environments
    for env, _ in envs_eval.values():
        env.close()

    return rank


def try_policy_instantiation():
    Policy()


def run(
    env, datastore: "DataStore", env_name: str, policy: "Policy", config: Dict[str, Any]
):
    # Instantiate metric for score calculation.
    metric = Metric(env_name=env_name, agent_names=datastore.agent_names)

    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)
            metric.store(infos=datastore.data["infos"], dones=datastore.data["dones"])

    return metric.results()


def to_codalab_scores_string(self) -> str:
    """Convert the data in scores to a CodaLab-scores-compatible string."""
    # NOTE: The score string names must be the same as in the competition.yaml.
    return (
        f"completion: {rank['completion']}\n"
        f"time: {rank['time']}\n"
        f"humanness: {rank['humanness']}\n"
        f"rules: {rank['rules']}\n"
    )


def resolve_codalab_dirs(
    root_path: str, input_dir: str = None, output_dir: str = None, local: bool = False,
) -> Tuple[str, str, str]:
    """Returns directories needed for the completion of the evaluation submission.

    Args:
        root_path (str): The path to the calling file.
        input_dir (str): The path containing the "res" and "ref" directories provided by
            CodaLab.
        output_dir (str): The path to output the scores.txt file.
        local (bool): If local directories should be used.

    Returns:
        Tuple[str, str, str]: The submission, evaluation-scenarios, and the scores directory,
            respectively. The submission directory contains the user submitted files,
            the evaluation scenarios directory contains the contents of the unzipped
            evaluation scenarios, and the scores directory is the directory in which
            to write the scores.txt file that is used to update the leaderboard.
    """
    logger.info(f"root_path={root_path}")
    logger.info(f"input_dir={input_dir}")
    logger.info(f"output_dir={output_dir}")

    if not local:
        submission_dir = os.path.join(input_dir, "res")
        evaluation_dir = os.path.join(input_dir, "ref")
    else:
        submission_dir = input_dir
        evaluation_dir = root_path
    scores_dir = output_dir

    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={submission_dir}")
    logger.info(f"evaluation_dir={evaluation_dir}")
    logger.info(f"scores_dir={scores_dir}")

    if not os.path.isdir(submission_dir):
        logger.warning(f"submission_dir={submission_dir} does not exist.")

    return submission_dir, evaluation_dir, scores_dir


def write_scores(scores, output_dir):
    if output_dir:
        with open(os.path.join(output_dir, _SCORES_FILENAME), "w") as output_file:
            output_file.write(scores)


def load_config_yaml(path: Path) -> Optional[Dict[str, Any]]:
    task_config = None
    if path.exists():
        with open(path, "r") as task_file:
            task_config = yaml.safe_load(task_file)
    return task_config


def resolve_config(
    base_config: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    # Use the default if None or empty.
    if not base_config:
        return defaults
    # Otherwise merge the two with the user config winning the tiebreaker
    return {**defaults, **base_config}


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
    args = parser.parse_args()

    # Get directories and install requirements.
    submit_dir, evaluation_dir, scores_dir = resolve_codalab_dirs(
        os.path.dirname(__file__), args.input_dir, args.output_dir, args.local
    )

    req_file = os.path.join(submit_dir, "requirements.txt")
    sys.path.insert(0, submit_dir)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-4",
        ]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

    import gym
    import yaml

    from copy_data import CopyData, DataStore
    from metric import Metric
    from score import Score

    from policy import Policy, submitted_wrappers

    evaluation_config = resolve_config(
        load_config_yaml(Path(evaluation_dir) / "config.yaml"),
        _DEFAULT_EVALUATION_CONFIG,
    )
    submission_config = resolve_config(
        load_config_yaml(Path(submit_dir) / "config.yaml"), _DEFAULT_SUBMISSION_CONFIG
    )

    unaccepted_keys = {*evaluation_config.keys()} - _EVALUATION_CONFIG_KEYS
    assert (
        len(unaccepted_keys) == 0
    ), f"Unaccepted evaluation config keys: {unaccepted_keys}"

    # Skip this if there is no evaluation
    if evaluation_config["evaluate"]:
        # Evaluate and write score.
        rank = evaluate(evaluation_config, submission_config)
        write_scores(to_codalab_scores_string(rank), args.output_dir)
    elif evaluation_config["validate"]:
        try_policy_instantiation()
