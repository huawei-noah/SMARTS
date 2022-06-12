from typing import Any, Dict, List

import gym
from submission.policy import submitted_policy, submitted_wrappers


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
        wrappers=wrappers,
        action_space="Continuous",
    )

    # Wrap the environment
    for wrapper in wrappers:
        env = wrapper(env)

    return env


def run(env, policy):
    total_return = 0.0
    for _ in range(config["eval"]["episodes"]):
        time_step = env.reset()
        ep_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            ep_return += time_step.reward

        # print(f"Eval episode {ep} return: {ep_return.numpy()[0]:.2f}")
        total_return += ep_return

    avg_return = total_return / config["eval"]["episodes"]
    print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Evaluating. Episode average return: {avg_return.numpy()[0]:.2f}")

    return


def evaluate(policy, wrappers):   
    # Make training and evaluation environments.
    envs_eval = {}
    for scen in config["scenarios"]:
        envs_eval[f"{scen}"] = make_env(
            config=config, scenario=scen, wrappers=wrappers
        )

    # Evaluate model for each scenario
    for env_name, env_eval in envs_eval.items():
        print(f"Evaluating env {env_name}.")
        run(env_eval=env_eval, config=config)
    print("\nFinished evaluating.\n")

    # Close all environments
    for env in envs_eval.values():
        env.close()


if __name__ == "__main__":
  
 
    evaluate(submitted_policy(), submitted_wrappers())
