import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__file__)

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
        sumo_headless=False,
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
    for index, (env_name, (env, datastore, seed)) in enumerate(envs_eval.items()):
        logger.info(f"\n{index}. Evaluating env {env_name}.\n")
        run(
            env=env,
            datastore=datastore,
            env_name=env_name,
            policy=policy,
            config=config,
            seed=seed,
        )
       
    # Close all environments
    for env, _, _ in envs_eval.values():
        env.close()



def run(
    env,
    datastore: "DataStore",
    env_name: str,
    policy: "Policy",
    config: Dict[str, Any],
    seed: Optional[int],
):
    # Instantiate metric for score calculation.

    # Ensure deterministic seeding
    env.seed((seed or 0) + config["seed"])
    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)
    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(prog="codalab-evaluation")
    # parser.add_argument(
    #     "--input_dir",
    #     help=(
    #         "The path to the directory containing the reference data and user "
    #         "submission data."
    #     ),
    #     required=True,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     help=(
    #         "Path to the directory where the submission's scores.txt file will be "
    #         "written to."
    #     ),
    #     required=True,
    #     type=str,
    # )
    # parser.add_argument(
    #     "--local",
    #     help="Flag to set when running evaluate locally. Defaults to False.",
    #     action="store_true",
    # )
    # args = parser.parse_args()

    # Get directories.
    # from utils import resolve_codalab_dirs

    # root_path = str(Path(__file__).absolute().parent)
    # submit_dir, evaluation_dir, scores_dir = resolve_codalab_dirs(
    #     root_path=root_path,
    #     input_dir=args.input_dir,
    #     # output_dir=args.output_dir,
    #     local=args.local,
    # )
   
    # req_file = os.path.join(submit_dir, "requirements.txt")
    # sys.path.insert(0, submit_dir) 

    import gym
    from copy_data import CopyData, DataStore
    from policy import Policy, submitted_wrappers
    config = {'phase': 'track1', 'eval_episodes': 50, 'seed': 42, 
              'scenarios': [
                            # "1_to_2lane_left_turn_c",
                            # "1_to_2lane_left_turn_t",
                            # "3lane_merge_multi_agent",
                            # "3lane_merge_single_agent",
                            # "3lane_cruise_multi_agent",
                            # "3lane_cruise_single_agent",
                            "3lane_cut_in",
                            "3lane_overtake",], 
              'bubble_env_evaluation_seeds': [], 'img_meters': 64, 'img_pixels': 256}

    evaluate(config)