import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common


def dqn(env, q_net, config):
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
    )

    agent.initialize()
