import gym
import sys
from pathlib import Path
from typing import Any, Dict

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[1]))

from submission.policy import Policy, submitted_wrappers


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
        # Wrap the environment
        for wrapper in submitted_wrappers():
            env = wrapper(env)
        envs_eval[f"{scenario}"] = env

    # Instantiate submitted policy.
    policy = Policy()

    # Evaluate model for each scenario
    for index, (env_name, env) in enumerate(envs_eval.items()):
        print(f"\n{index}. Evaluating env {env_name}.\n")
        run(
            env=env,
            policy=policy,
            config=config,
        )

    # Close all environments
    for env in envs_eval.values():
        env.close()


def run(
    env,
    policy: "Policy",
    config: Dict[str, Any],
):
    env.seed(config["seed"])
    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":
    config = {
        "eval_episodes": 5,
        "seed": 42,
        "scenarios": [
            # "1_to_2lane_left_turn_c",
            # "1_to_2lane_left_turn_t",
            # "3lane_merge_multi_agent",
            # "3lane_merge_single_agent",
            "3lane_cruise_multi_agent",
            # "3lane_cruise_single_agent",
            # "3lane_cut_in",
            # "3lane_overtake",
        ],
        "img_meters": 64,
        "img_pixels": 256,
    }

    evaluate(config)
