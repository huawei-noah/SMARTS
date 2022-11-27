import gym
import sys
from pathlib import Path
from ruamel.yaml import YAML
from typing import Any, Dict
yaml = YAML(typ="safe")

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submission.policy import Policy, submitted_wrappers
from submission import network

def evaluate(config):
    # Make evaluation environments.
    envs_eval = {}
    for scenario in config["scenarios"]:
        env = gym.make(
            "smarts.env:multi-scenario-v0",
            scenario=scenario,
            action_space="TargetPose",
            img_meters=int(config["img_meters"]),
            img_pixels=int(config["img_pixels"]),
            sumo_headless=config["sumo_headless"],
            headless=config["headless"],
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
        run(env=env, policy=policy, config=config)

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
    # Load config file.
    config_file = yaml.load(
        (Path(__file__).resolve().parents[0] / "config.yaml").read_text()
    )
    config = {
        "eval_episodes": 5,
        "seed": 42,
        "scenarios": [
            # "1_to_2lane_left_turn_c",
            # "1_to_2lane_left_turn_t",
            "3lane_merge_multi_agent",
            # "3lane_merge_single_agent",
            # "3lane_cruise_multi_agent",
            # "3lane_cruise_single_agent",
            # "3lane_cut_in",
            # "3lane_overtake",
        ],
        "img_meters": config_file["smarts"]["img_meters"],
        "img_pixels": config_file["smarts"]["img_pixels"],
        "sumo_headless": False,
        "headless": True,
    }

    evaluate(config)
