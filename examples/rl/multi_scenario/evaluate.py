from typing import Any, Dict, List

import gym
from submission.policy import Policy, submitted_wrappers
from submission.policy import IMG_METERS, IMG_PIXELS


def make_env(
    config: Dict[str, Any], scenario: str, wrappers: List[gym.Wrapper] = []
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
    )

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def evaluate():
    config = {
        "img_meters": IMG_METERS,
        "img_pixels": IMG_PIXELS,
        "eval_episodes": 1e3,
    }
    scenarios = [
        "1_to_2lane_left_turn_c",
        "1_to_2lane_left_turn_t",
        "3lane_merge_multi_agent",
        "3lane_merge_single_agent",
        "3lane_cruise_multi_agent"
        "3lane_cruise_single_agent",
        "3lane_cut_off",
        "3lane_overtake",
    ]

    # Make evaluation environments.
    envs_eval = {}
    for scen in scenarios:
        envs_eval[f"{scen}"] = make_env(config=config, scenario=scen, wrappers=submitted_wrappers())

    policy = Policy()

    # Evaluate model for each scenario
    for env_name, env_eval in envs_eval.items():
        print(f"Evaluating env {env_name}.")
        run(env=env_eval, policy=policy, config=config)
    print("\nFinished evaluating.\n")

    # Close all environments
    for env in envs_eval.values():
        env.close()


def run(env, policy, config):
    total_return = 0.0
    for _ in range(config["eval_episodes"]):
        observations = env.reset()
        ep_return = 0.0
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.action(observations)
            observations, rewards, dones, infos = env.step(actions)
            for 
            collisions += infos[agent_id]
            ep_return += rewards.reward

        # print(f"Eval episode {ep} return: {ep_return.numpy()[0]:.2f}")
        total_return += ep_return

    avg_return = total_return / config["eval"]["episodes"]
    print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Evaluating. Episode average return: {avg_return.numpy()[0]:.2f}")

    return

class Metric:
    def __init__(self):

    def compute(self, infos):
        for infos.items()


if __name__ == "__main__":

    evaluate()
