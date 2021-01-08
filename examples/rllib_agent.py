from pathlib import Path

import gym
import numpy as np

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.env.custom_observations import lane_ttc_observation_adapter

tf = try_import_tf()

# This action space should match the input to the action_adapter(..) function below.
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)


# This observation space should match the output of observation_adapter(..) below
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


def reward_adapter(env_obs, env_reward):
    return env_reward


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering * np.pi * 0.25])


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class RLLibTFSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space):
        path_to_model = str(path_to_model)  # might be a str or a Path, normalize to str
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        tf.compat.v1.saved_model.load(
            self._sess, export_dir=path_to_model, tags=["serve"]
        )
        self._output_node = self._sess.graph.get_tensor_by_name("default_policy/add:0")
        self._input_node = self._sess.graph.get_tensor_by_name(
            "default_policy/observation:0"
        )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        obs = self._prep.transform(obs)
        # These tensor names were found by inspecting the trained model
        res = self._sess.run(self._output_node, feed_dict={self._input_node: [obs]})
        action = res[0]
        return action


rllib_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=500),
        agent_params={
            "path_to_model": Path(__file__).resolve().parent / "model",
            "observation_space": OBSERVATION_SPACE,
        },
        agent_builder=RLLibTFSavedModelAgent,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}
