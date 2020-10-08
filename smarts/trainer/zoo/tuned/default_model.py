"""
This file contain default network for rllib training,
and can be used for policy evaluation
"""
import pickle

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf

from smarts.core.agent import AgentPolicy

tf = try_import_tf()


# actually, i think here define Model is useless since that the RLlib use fcnet_v2 as default
# #See: https://github.com/ray-project/ray/blob/b89cac976ae171d6d9b3245394e4932288fc6f11/rllib/models/tf/fcnet_v2.py#L14
class Model(FullyConnectedNetwork):
    NAME = "model"


ModelCatalog.register_custom_model(Model.NAME, Model)


class RLLibTFCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        load_path = str(load_path)
        if algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy as LoadPolicy
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.Session(graph=tf.Graph())

        with tf.name_scope(policy_name):
            # Observation space needs to be flattened before passed to the policy
            flat_obs_space = self._prep.observation_space
            policy = LoadPolicy(flat_obs_space, action_space, {})
            objs = pickle.load(open(load_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[policy_name]
            policy.set_weights(weights)

        # These tensor names were found by inspecting the trained model
        if algorithm == "PPO":
            # CRUCIAL FOR SAFETY:
            #   We use Tensor("split") instead of Tensor("add") to force
            #   PPO to be deterministic.
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observation:0"
            )
            self._output_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/split:0"
            )
        elif algorithm == "DQN":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/value_out/BiasAdd:0"
                ),
                axis=1,
            )
        else:
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(f"{policy_name}/fc_out/BiasAdd:0"),
                axis=1,
            )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        obs = self._prep.transform(obs)
        res = self._sess.run(self._output_node, feed_dict={self._input_node: [obs]})
        action = res[0]
        return action


class RLLibTFSavedModelPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space):
        load_path = str(load_path)
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        tf.saved_model.load(
            self._sess, export_dir=load_path, tags=["serve"], clear_devices=True,
        )
        # These tensor names were found by inspecting the trained model
        if algorithm == "PPO":
            # CRUCIAL FOR SAFETY:
            #   We use Tensor("split") instead of Tensor("add") to force
            #   PPO to be deterministic.
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observation:0"
            )
            self._output_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/split:0"
            )
        # todo: need to check
        elif algorithm == "DQN":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/value_out/BiasAdd:0"
                ),
                axis=1,
            )
        else:
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(f"{policy_name}/fc_out/BiasAdd:0"),
                axis=1,
            )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        obs = self._prep.transform(obs)
        res = self._sess.run(self._output_node, feed_dict={self._input_node: [obs]})
        action = res[0]
        return action


class BatchRLLibTFCheckpointPolicy(AgentPolicy):
    def __init__(
        self, load_path, algorithm, policy_name, observation_space, action_space
    ):
        load_path = str(load_path)
        if algorithm == "PPO":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif algorithm in ["A2C", "A3C"]:
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
        elif algorithm == "PG":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif algorithm == "DQN":
            from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy as LoadPolicy
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.Session(graph=tf.Graph())

        with tf.name_scope(policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(load_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[policy_name]
            policy.set_weights(weights)

        # These tensor names were found by inspecting the trained model
        if algorithm == "PPO":
            # CRUCIAL FOR SAFETY:
            #   We use Tensor("split") instead of Tensor("add") to force
            #   PPO to be deterministic.
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observation:0"
            )
            self._output_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/split:0"
            )
        elif self._algorithm == "DQN":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/value_out/BiasAdd:0"
                ),
                axis=1,
            )
        else:
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(f"{policy_name}/fc_out/BiasAdd:0"),
                axis=1,
            )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        agent_id = list(obs.keys())
        obs = list(obs.values())
        obs = [self._prep.transform(o) for o in obs]
        res = self._sess.run(self._output_node, feed_dict={self._input_node: obs})
        actions = res
        actions = dict(zip(agent_id, actions))
        return actions


class BatchRLLibTFSavedModelPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, observation_space):
        load_path = str(load_path)
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.Session(graph=tf.Graph())
        tf.saved_model.load(
            self._sess, export_dir=load_path, tags=["serve"], clear_devices=True,
        )
        # These tensor names were found by inspecting the trained model
        if algorithm == "PPO":
            # CRUCIAL FOR SAFETY:
            #   We use Tensor("split") instead of Tensor("add") to force
            #   PPO to be deterministic.
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observation:0"
            )
            self._output_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/split:0"
            )
        elif algorithm == "DQN":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/value_out/BiasAdd:0"
                ),
                axis=1,
            )
        else:
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(f"{policy_name}/fc_out/BiasAdd:0"),
                axis=1,
            )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        agent_id = list(obs.keys())
        obs = [self._prep.transform(o) for o in obs.values()]
        res = self._sess.run(self._output_node, feed_dict={self._input_node: obs})
        # iterating over a dictionary is guaranteed to be in a deterministic order
        # so it's safe to zip here.
        actions = dict(zip(agent_id, res))
        return actions
