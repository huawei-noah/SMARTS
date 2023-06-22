from pathlib import Path

import gymnasium as gym
import numpy as np
from ray.rllib.utils.typing import ModelConfigDict

# ray[rllib] is not the part of main dependency of the SMARTS package. It needs to be installed separately
# as a part of the smarts[train] dependency using the command "pip install -e .[train]. The following try block checks
# whether ray[rllib] was installed by user and raises an Exception warning the user to install it if not so.
try:
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
    from ray.rllib.utils import try_import_tf
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib_agent.py")


from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.custom_observations import lane_ttc_observation_adapter
from smarts.zoo.agent_spec import AgentSpec

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

# The FullyConnectedNetwork expects a flattened space.
FLATTENED_OBSERVATION_SPACE = gym.spaces.utils.flatten_space(OBSERVATION_SPACE)


def observation_adapter(agent_observation, /):
    return gym.spaces.utils.flatten(
        OBSERVATION_SPACE, lane_ttc_observation_adapter.transform(agent_observation)
    )


def action_adapter(agent_action, /):
    throttle, brake, steering = agent_action
    return np.array([throttle, brake, steering * np.pi * 0.25], dtype=np.float32)


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return super().forward(input_dict, state, seq_lens)


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class RLLibTFSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space, policy_name="default_policy"):
        path_to_model = str(path_to_model)  # might be a str or a Path, normalize to str
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        tf.compat.v1.saved_model.load(
            self._sess, export_dir=path_to_model, tags=["serve"]
        )
        self._output_node = self._sess.graph.get_tensor_by_name(f"policy_name/add:0")
        self._input_node = self._sess.graph.get_tensor_by_name(
            f"{policy_name}/observation:0"
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
            "observation_space": FLATTENED_OBSERVATION_SPACE,
        },
        agent_builder=RLLibTFSavedModelAgent,
        observation_adapter=observation_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": FLATTENED_OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}
