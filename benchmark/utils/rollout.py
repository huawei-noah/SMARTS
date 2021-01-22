import collections

from ray import logger
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray


def rollout(trainer, env_name, metrics_handler, num_steps, num_episodes, log_dir):
    """Reference: https://github.com/ray-project/ray/blob/master/rllib/rollout.py"""
    policy_agent_mapping = default_policy_agent_mapping
    assert hasattr(trainer, "workers") and isinstance(trainer.workers, WorkerSet)
    env = trainer.workers.local_worker().env
    multiagent = isinstance(env, MultiAgentEnv)
    if trainer.workers.local_worker().multiagent:
        policy_agent_mapping = trainer.config["multiagent"]["policy_mapping_fn"]
    policy_map = trainer.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    for episode in range(num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        reward_total = 0.0
        step = 0
        while not done and step < num_steps:
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = trainer.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = trainer.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)

            metrics_handler.log_step(
                episode=episode,
                observations=multi_obs,
                actions=action,
                rewards=reward,
                dones=done,
                infos=info,
            )

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            # filter dead agents
            if multiagent:
                next_obs = {
                    agent_id: obs
                    for agent_id, obs in next_obs.items()
                    if not done[agent_id]
                }

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            step += 1
            obs = next_obs
        logger.info(
            "\nEpisode #{}: steps: {} reward: {}".format(episode, step, reward_total)
        )
        if done:
            episode += 1
    metrics_handler.write_to_csv(csv_dir=log_dir)
    if show_plots:
        metrics_handler.show_plots(**plot_kwargs)
