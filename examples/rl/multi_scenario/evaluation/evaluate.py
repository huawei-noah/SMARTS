from typing import Any, Dict, List

import gym
from ..evaluation.copy_data import CopyData, DataStore
from ..submission.policy import IMG_METERS, IMG_PIXELS, Policy, submitted_wrappers

def make_env(
    config: Dict[str, Any], scenario: str, datastore:DataStore, wrappers: List[gym.Wrapper] = [],
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

    # Makes a copy of original info and prevents modification by external users. 
    env = CopyData(env, datastore)

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
        "3lane_merge_single_agent",
        # "3lane_cruise_multi_agent",
        # "3lane_cruise_single_agent",
        # "3lane_cut_in",
        # "3lane_overtake",
    ]

    # Make evaluation environments.
    envs_eval = {}
    for scen in scenarios:
        datastore =  DataStore()
        envs_eval[f"{scen}"] = (make_env(
            config=config, scenario=scen, datastore=datastore, wrappers=submitted_wrappers(),
         ), datastore)

    # Instantiate submitted policy.
    policy = Policy()

    # Instantiate metric for score calculation.
    metric = Metric()

    # Evaluate model for each scenario
    for env_name, env_eval in envs_eval.items():
        print(f"Evaluating env {env_name}.")
        run(env=env_eval, name=env_name, policy=policy, config=config, metric=metric)
    print("\nFinished evaluating.\n")

    # Close all environments
    for env in envs_eval.values():
        env.close()


def run(env, name, policy, config, metric):
    for _ in range(config["eval_episodes"]):
        observations = env[0].reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = policy.act(observations)
            observations, rewards, dones, infos = env[0].step(actions)

            # print("Info from copy: ", env[1])
            print("Info from copy: ", env[1]["Agent_0"]["score"])

            import time
            time.sleep(1)

            # metric.compute(infos)
    return

class Metric:
    def __init__(self):
        results = {}
 
#     def compute(self, infos):
#         for agent_name, agent_info in infos.items():
#             # results[agent_name] = agent_info[obs.events
#             pass


#         import os
#         os.exit(2)

        #     collisions += infos[agent_id]
        #     ep_return += rewards.reward

        # total_return += ep_return

    # avg_return = total_return / config["eval"]["episodes"]
    # print(f"Evaluating. Episode average return: {avg_return.numpy()[0]:.2f}")


    # def compute_collisions():
        

    # def get(self):

    #     return 0
        #     {

        # }


if __name__ == "__main__":

    evaluate()
