import reverb
from tf_agents.replay_buffers import (
    py_hashed_replay_buffer,
    reverb_replay_buffer,
    reverb_utils,
    tf_uniform_replay_buffer,
)
from tf_agents.specs import tensor_spec


def reverb_replay(env, agent, config):
    table_name = "uniform_table"
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=config["buffer_kwargs"]["replay_buffer_max_length"],
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server,
    )

    replay_buffer_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client, table_name, sequence_length=2
    )

    return replay_buffer, replay_buffer_observer


def uniform_replay(env, agent, config):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        max_length=config["buffer_kwargs"]["max_length"],
    )
    replay_buffer_observer = replay_buffer.add_batch

    return replay_buffer, replay_buffer_observer


def hashed_replay(env, agent, config):
    replay_buffer = py_hashed_replay_buffer.PyHashedReplayBuffer(
        data_spec=agent.collect_data_spec,
        capacity=config["buffer_kwargs"]["capacity"],
        log_interval=None,
    )
    replay_buffer_observer = replay_buffer.add_batch

    return replay_buffer, replay_buffer_observer
