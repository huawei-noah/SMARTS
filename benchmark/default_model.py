"""
This file contain default network for rllib training,
and can be used for policy evaluation
"""
import pickle
import tensorflow as tf

from pathlib import Path

from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.trainer import with_common_config

from smarts.core.agent import AgentPolicy

from benchmark.agents import load_config

tf1, tf, tfv = try_import_tf()


BASE_DIR = Path(__file__).expanduser().absolute().parent.parent


class RLLibTFCheckpointPolicy(AgentPolicy):
    def __init__(self, load_path, algorithm, policy_name, yaml_path):
        load_path = str(load_path)
        if algorithm == "ppo":
            from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
        elif algorithm in "a2c":
            from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy as LoadPolicy
            from ray.rllib.agents.a3c import DEFAULT_CONFIG
        elif algorithm == "pg":
            from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy as LoadPolicy
        elif algorithm == "dqn":
            from ray.rllib.agents.dqn import DQNTFPolicy as LoadPolicy
        elif algorithm == "maac":
            from benchmark.agents.maac.tf_policy import CA2CTFPolicy as LoadPolicy
            from benchmark.agents.maac.tf_policy import DEFAULT_CONFIG
        elif algorithm == "maddpg":
            from benchmark.agents.maddpg.tf_policy import MADDPG2TFPolicy as LoadPolicy
            from benchmark.agents.maddpg.tf_policy import DEFAULT_CONFIG
        elif algorithm == "mfac":
            from benchmark.agents.mfac.tf_policy import MFACTFPolicy as LoadPolicy
            from benchmark.agents.mfac.tf_policy import DEFAULT_CONFIG
        elif algorithm == "networked_pg":
            from benchmark.agents.networked_pg.tf_policy import (
                NetworkedPG as LoadPolicy,
            )
            from benchmark.agents.networked_pg.tf_policy import (
                PG_DEFAULT_CONFIG as DEFAULT_CONFIG,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        yaml_path = BASE_DIR / yaml_path
        load_path = BASE_DIR / f"log/results/run/{load_path}"

        config = load_config(yaml_path)
        observation_space = config["policy"][1]
        action_space = config["policy"][2]
        pconfig = DEFAULT_CONFIG

        pconfig["model"].update(config["policy"][-1].get("model", {}))
        pconfig["agent_id"] = policy_name

        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.Session(graph=tf.get_default_graph())

        with tf.name_scope(policy_name):
            # Observation space needs to be flattened before passed to the policy
            flat_obs_space = self._prep.observation_space
            policy = LoadPolicy(flat_obs_space, action_space, pconfig)
            self._sess.run(tf.global_variables_initializer())
            objs = pickle.load(open(load_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[policy_name]
            policy.set_weights(weights)

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        # These tensor names were found by inspecting the trained model
        if algorithm == "ppo":
            # CRUCIAL FOR SAFETY:
            #   We use Tensor("split") instead of Tensor("add") to force
            #   PPO to be deterministic.
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observation:0"
            )
            self._output_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/split:0"
            )
        elif algorithm == "dqn":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/observations:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/value_out/BiasAdd:0"
                ),
                axis=1,
            )
        elif algorithm == "maac":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/policy-inputs:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/logits_out/BiasAdd:0"
                ),
                axis=1,
            )
        elif algorithm == "maddpg":
            self._input_node = self._sess.graph.get_tensor_by_name(
                f"{policy_name}/obs_2:0"
            )
            self._output_node = tf.argmax(
                self._sess.graph.get_tensor_by_name(
                    f"{policy_name}/actor/AGENT_2_actor_RelaxedOneHotCategorical_1/sample/AGENT_2_actor_exp/forward/Exp:0"
                )
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
