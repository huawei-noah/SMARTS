import argparse
import ray
import collections
import gym

from ray import logger
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray

from benchmark import gen_config
from benchmark.metrics import basic_metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser("Run evaluation")
    parser.add_argument(
        "scenario", type=str, help="Scenario name",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=10)
    # TODO(ming): eliminate this arg
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument("--config_file", "-f", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./log/results")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main(
    scenario,
    config_file,
    checkpoint,
    log_dir,
    num_steps=1000,
    num_episodes=10,
    paradigm="decentralized",
    headless=False,
    show_plots=False,
):

    config = gen_config(
        scenario=scenario,
        config_file=config_file,
        checkpoint=checkpoint,
        num_steps=num_steps,
        num_episodes=num_episodes,
        paradigm=paradigm,
        headless=headless,
    )

    ray.init()
    tune_config = config["run"]["config"]
    trainer_cls = config["trainer"]
    trainer_config = {"env_config": config["env_config"]}
    if paradigm != "centralized":
        trainer_config.update({"multiagent": tune_config["multiagent"]})
    else:
        trainer_config.update({"model": tune_config["model"]})

    trainer = trainer_cls(env=tune_config["env"], config=trainer_config)

    trainer.restore(checkpoint)
    rollout(trainer, None, num_steps, num_episodes, show_plots)
    trainer.stop()


def rollout(trainer, env_name, num_steps, num_episodes, show_plots):
    """Reference: https://github.com/ray-project/ray/blob/master/rllib/rollout.py"""
    policy_agent_mapping = default_policy_agent_mapping
    if hasattr(trainer, "workers") and isinstance(trainer.workers, WorkerSet):
        env = trainer.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if trainer.workers.local_worker().multiagent:
            policy_agent_mapping = trainer.config["multiagent"]["policy_mapping_fn"]

        policy_map = trainer.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: trainer.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(trainer)
            )
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    metrics_handler = metrics.MetricHandler(num_episodes)

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

            metrics_handler.log_step(multi_obs, reward, done, info, episode=episode)

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
    metrics_handler.write_to_csv(csv_dir=args.log_dir)
    if show_plots:
        metrics_handler.show_plots()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        paradigm=args.paradigm,
        headless=args.headless,
        show_plots=args.plot,
        log_dir=args.log_dir,
    )
