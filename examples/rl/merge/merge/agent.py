import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent


def dqn(env, network, config):
    train_step_counter = tf.Variable(0, dtype=tf.int64)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["agent_kwargs"]["learning_rate"]
    )

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["agent_kwargs"]["epsilon_greedy"][
            "initial"
        ],  # initial ε
        decay_steps=config["agent_kwargs"]["epsilon_greedy"]["decay_steps"],
        end_learning_rate=config["agent_kwargs"]["epsilon_greedy"]["end"],  # final ε
    )

    agent = DqnAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=env.action_spec(),
        q_network=network,
        optimizer=optimizer,
        target_update_period=config["agent_kwargs"]["target_update_period"],
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,  # discount factor
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        n_step_update=config["agent_kwargs"]["n_step_update"],
    )

    agent.initialize()

    return agent
