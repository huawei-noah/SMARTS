from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np

try:
    from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.evaluation.episode import Episode
    from ray.rllib.evaluation.episode_v2 import EpisodeV2
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils.typing import PolicyID
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib.py")

import smarts
from smarts.sstudio.scenario_construction import build_scenario

if __name__ == "__main__":
    from configs import gen_parser, gen_pg_config
    from rllib_agent import TrainingModel, rllib_agent
else:
    from .configs import gen_parser, gen_pg_config
    from .rllib_agent import TrainingModel, rllib_agent

# Add custom metrics to your tensorboard using these callbacks
# See: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        episode.user_data["ego_reward"] = []

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        single_agent_id = list(episode.get_agents())[0]
        infos = episode._last_infos.get(single_agent_id)
        if infos is not None:
            episode.user_data["ego_reward"].append(infos["reward"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        mean_ego_speed = np.mean(episode.user_data["ego_reward"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_reward={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_reward"] = mean_ego_speed


def main(
    scenario,
    envision,
    time_total_s,
    rollout_fragment_length,
    train_batch_size,
    seed,
    num_samples,
    num_agents,
    num_workers,
    resume_training,
    result_dir,
    checkpoint_freq: int,
    checkpoint_num: Optional[int],
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    save_model_path,
):
    agent_values = {
        "agent_specs": {
            f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)
        },
        "rllib_policies": {
            f"AGENT-{i}": (
                None,
                rllib_agent["observation_space"],
                rllib_agent["action_space"],
                {"model": {"custom_model": TrainingModel.NAME}},
            )
            for i in range(num_agents)
        },
    }
    rllib_policies = agent_values["rllib_policies"]
    agent_specs = agent_values["agent_specs"]

    smarts.core.seed(seed)
    algo_config: AlgorithmConfig = gen_pg_config(
        scenario=scenario,
        envision=envision,
        rollout_fragment_length=rollout_fragment_length,
        train_batch_size=train_batch_size,
        num_workers=num_workers,
        seed=seed,
        log_level=log_level,
        rllib_policies=rllib_policies,
        agent_specs=agent_specs,
        callbacks=Callbacks,
    )

    def get_checkpoint_dir(num):
        checkpoint_dir = result_dir / f"checkpoint_{num}" / f"checkpoint-{num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    if resume_training:
        checkpoint = str(get_checkpoint_dir("latest"))
    if checkpoint_num:
        checkpoint = str(get_checkpoint_dir(checkpoint_num))
    else:
        checkpoint = None

    print(f"======= Checkpointing at {str(result_dir)} =======")

    algo = algo_config.build()
    if checkpoint is not None:
        Algorithm.load_checkpoint(algo, checkpoint=checkpoint)
    result = {}
    current_iteration = 0
    checkpoint_iteration = checkpoint_num or 0

    try:
        while result.get("time_total_s", 0) < time_total_s:
            result = algo.train()
            print(f"======== Iteration {result['training_iteration']} ========")
            print(result, depth=1)

            if current_iteration % checkpoint_freq == 0:
                checkpoint_dir = get_checkpoint_dir(checkpoint_iteration)
                print(f"======= Saving checkpoint {checkpoint_iteration} =======")
                algo.save_checkpoint(checkpoint_dir)
                checkpoint_iteration += 1
            current_iteration += 1
        algo.save_checkpoint(get_checkpoint_dir(checkpoint_iteration))
    finally:
        algo.save_checkpoint(get_checkpoint_dir("latest"))
        algo.stop()


if __name__ == "__main__":
    default_save_model_path = str(
        Path(__file__).expanduser().resolve().parent / "pg_model"
    )
    default_result_dir = str(Path(__file__).resolve().parent / "results" / "pg_results")
    parser = gen_parser("rllib-example", default_result_dir, default_save_model_path)

    args = parser.parse_args()
    build_scenario(scenario=args.scenario, clean=False, seed=42)

    main(
        scenario=args.scenario,
        envision=args.envision,
        time_total_s=args.time_total_s,
        rollout_fragment_length=args.rollout_fragment_length,
        train_batch_size=args.train_batch_size,
        seed=args.seed,
        num_samples=args.num_samples,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        resume_training=args.resume_training,
        result_dir=args.result_dir,
        checkpoint_freq=max(args.checkpoint_freq, 1),
        checkpoint_num=args.checkpoint_num,
        log_level=args.log_level,
        save_model_path=args.save_model_path,
    )
