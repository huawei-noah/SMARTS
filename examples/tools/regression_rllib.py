import json
import math
import multiprocessing
import os
import tempfile
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from examples.rllib_agent import TrainingModel
from smarts.core.agent import Agent, AgentSpec
from smarts.env.custom_observations import lane_ttc_observation_adapter
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent_interface import AgentInterface, AgentType

HORIZON = 5000

tf = try_import_tf()


class RLlibTFSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space):
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._path_to_model = path_to_model

    def setup(self):
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._sess.__enter__()
        tf.compat.v1.saved_model.load(
            self._sess, export_dir=self._path_to_model, tags=["serve"]
        )

    def act(self, obs):
        obs = self._prep.transform(obs)
        graph = tf.compat.v1.get_default_graph()
        # These tensor names were found by inspecting the trained model
        output_node = graph.get_tensor_by_name("default_policy/add:0")
        input_node = graph.get_tensor_by_name("default_policy/observation:0")
        res = self._sess.run(output_node, feed_dict={input_node: [obs]})
        action = res[0]
        return action


ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)


OBSERVATION_SPACE = lane_ttc_observation_adapter.space


def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


def run_experiment(log_path, experiment_name, training_iteration=100):
    model_path = Path(__file__).parent / "model"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=5000),
        policy=RLlibTFSavedModelAgent(
            model_path.absolute(),
            OBSERVATION_SPACE,
        ),
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    )

    rllib_policies = {
        "policy": (
            None,
            OBSERVATION_SPACE,
            ACTION_SPACE,
            {"model": {"custom_model": TrainingModel.NAME}},
        )
    }

    scenario_path = Path(__file__).parent / "../../scenarios/loop"
    scenario_path = str(scenario_path.absolute())

    tune_confg = {
        "env": RLlibHiWayEnv,
        "env_config": {
            "scenarios": [scenario_path],
            "seed": 42,
            "headless": True,
            "agent_specs": {"Agent-007": agent_spec},
        },
        "multiagent": {
            "policies": rllib_policies,
            "policy_mapping_fn": lambda _: "policy",
        },
        "log_level": "WARN",
        "num_workers": multiprocessing.cpu_count() - 1,
        "horizon": HORIZON,
    }

    analysis = tune.run(
        "PPO",
        name=experiment_name,
        stop={"training_iteration": training_iteration},
        max_failures=10,
        local_dir=log_path,
        config=tune_confg,
    )

    return analysis


def create_df(file_path):
    data = {}
    with open(file_path, encoding="utf-8", errors="ignore") as json_data:
        for i, r in enumerate(json_data.readlines()):
            data[i] = json.loads(r)
    df = pd.DataFrame.from_dict(data, orient="index")
    return df


def main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Change these consts if needed
        experiments_count = 10
        iteration_times = 100
        experiment_name = "learning_regression_test"

        for i in range(experiments_count):
            run_experiment(tmpdirname, experiment_name, iteration_times)

        p = Path(os.path.join(tmpdirname, experiment_name))

        data_frames = []  # data frame objects of these experiments
        for d in p.iterdir():
            if d.is_dir():
                f = d / "result.json"
                if f.exists():
                    data_frames.append(create_df(f.absolute()))

        df_experiments = pd.concat(tuple(data_frames)).groupby(level=0)
        mean_reward_stats = df_experiments["episode_reward_mean"].agg(
            ["mean", "count", "std"]
        )

        # Only ci95_lo will be used
        ci95_hi = []
        ci95_lo = []
        for i in mean_reward_stats.index:
            m, c, s = mean_reward_stats.loc[i]
            ci95_hi.append(m + 1.96 * s / math.sqrt(c))
            ci95_lo.append(m - 1.96 * s / math.sqrt(c))
        mean_reward_stats["ci95_hi"] = ci95_hi
        mean_reward_stats["ci95_lo"] = ci95_lo

        print("CI95_REWARD_MEAN:", ci95_lo[iteration_times - 1])
        ci95_file = Path(__file__).parent / "../../smarts/env/tests/ci95_reward_lo"
        with ci95_file.open("w+") as f:
            f.write(str(ci95_lo[iteration_times - 1]))


if __name__ == "__main__":
    main()
