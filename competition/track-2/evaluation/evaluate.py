import os
import sys
import subprocess
import argparse
from typing import Any, Dict


_SCORES_FILENAME = "scores.txt"


def make_env(
    config: Dict[str, Any],
    scenario: str,
    datastore: "DataStore",
    wrappers=[],
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
        action_space="TargetPose",
        sumo_headless=True,
    )

    # Make a copy of original info.
    env = CopyData(env, datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment
    for wrapper in wrappers:
        #breakpoint()
        env = wrapper(env)

    return env


def evaluate():
    config = {
        "img_meters": IMG_METERS,
        "img_pixels": IMG_PIXELS,
        "eval_episodes": 2,
    }
    scenarios = [
        "1_to_2lane_left_turn_c",
        # "1_to_2lane_left_turn_t",
        # "3lane_merge_multi_agent",
        # "3lane_merge_single_agent",
        # "3lane_cruise_multi_agent",
        # "3lane_cruise_single_agent",
        # "3lane_cut_in",
        # "3lane_overtake",
    ]

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
        print(f"\n{index}. Evaluating env {env_name}.\n")
        counts, costs = run(
            env=env,
            datastore=datastore,
            env_name=env_name,
            policy=policy,
            config=config,
        )
        score.add(counts, costs)

    rank = score.compute()
    print("\nOverall Rank:\n", rank)
    print("\nFinished evaluating.\n")

    # Close all environments
    for env, _ in envs_eval.values():
        env.close()

    return rank


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


def write_scores(scores, output_dir):
    if output_dir:
        with open(os.path.join(output_dir, _SCORES_FILENAME), "w") as output_file:
            output_file.write(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="codalab-evaluation")
    parser.add_argument(
        "--input_dir",
        help=(
            "The path to the directory containing the reference data and user "
            "submission data."
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help=(
            "Path to the directory where the submission's scores.txt file will be "
            "written to."
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--local",
        help="Flag to set when running evaluate locally. Defaults to False.",
        action="store_true",
    )
    args = parser.parse_args()

    # Get directories and install requirements.
    if args.local:
        submit_dir = args.input_dir
    else:
        submit_dir = os.path.join(args.input_dir, "res")
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

    from copy_data import CopyData, DataStore
    from metric import Metric
    from score import Score
    from policy import IMG_METERS, IMG_PIXELS, Policy, submitted_wrappers

    # Evaluate and write score.
    rank = evaluate()
    write_scores(to_codalab_scores_string(rank), args.output_dir)
