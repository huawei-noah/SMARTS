# Add submission directory to python search path
import sys
from pathlib import Path

print(f"Adding python search path: {Path(__file__).absolute().parents[1]}")
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from typing import Any, Dict, List

import gym
from evaluation.copy_data import CopyData, DataStore
from evaluation.counter import Counter
from evaluation.metric import Metric, Score
from submission.policy import IMG_METERS, IMG_PIXELS, Policy, submitted_wrappers


def make_env(
    config: Dict[str, Any],
    scenario: str,
    datastore: DataStore,
    wrappers: List[gym.Wrapper] = [],
) -> gym.Env:
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
    )

    # Make a copy of original info.
    env = CopyData(env, datastore)
    # Disallow modification of attributes starting with "_" by external users.
    env = gym.Wrapper(env)

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def evaluate():
    config = {
        "img_meters": IMG_METERS,
        "img_pixels": IMG_PIXELS,
        "eval_episodes": 100,
    }
    scenarios = [
        # "1_to_2lane_left_turn_c",
        # "1_to_2lane_left_turn_t",
        "3lane_merge_multi_agent",
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
    for env_name, (env, datastore) in envs_eval.items():
        print(f"Evaluating env {env_name}.")
        metric = run(
            env=env, datastore=datastore, name=env_name, policy=policy, config=config
        )

        # score[]

    print("\nFinished evaluating.\n")

    # Close all environments
    for env, _ in envs_eval.values():
        env.close()


def run(env, datastore: DataStore, name, policy: Policy, config: Dict[str, Any]):
    # Instantiate metric for score calculation.
    metric = Metric(datastore.agent_names)
    steps = Counter()

    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            # Count steps per scenario
            if not observations:
                steps.increment()

            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)

            import time

            time.sleep(0.5)
            print(dones)

            # metric.store(datastore.data)
            # if dones["__all__"]:
            #     metric.compute()

    return


# things to check
# 1 cheating by changing datainfo
# 2 distance to obstacles
# 3 check whether waypoint for both lanes avaialble in left turn
# 4 check jd
# check score calculation


if __name__ == "__main__":
    evaluate()
