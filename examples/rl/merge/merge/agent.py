import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils import common


def dqn(env, q_net, config):
    train_step_counter = tf.Variable(0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["alg_kwargs"]["learning_rate"])
    # optimizer = tf.keras.optimizers.RMSprop(
    #     lr=2.5e-4, 
    #     rho=0.95, 
    #     momentum=0.0,
    #     epsilon=0.00001, 
    #     centered=True  
    # )
    
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config["alg_kwargs"]["epsilon_greedy"]["initial"], # initial ε
        decay_steps=config["alg_kwargs"]["epsilon_greedy"]["decay_steps"],
        end_learning_rate=config["alg_kwargs"]["epsilon_greedy"]["end"], # final ε
    )

    agent = DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda _: epsilon_fn(train_step_counter)
    )

    agent.initialize()

    return agent

update_period = 4 # train the model every 4 steps
agent = DqnAgent(
    target_update_period=2000, # <=> 32,000 ALE frames
    td_errors_loss_fn=keras.losses.Huber(reduction="none"),
    gamma=0.99, # discount factor
)
