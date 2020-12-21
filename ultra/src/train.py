import os
from ultra.utils.ray import default_ray_kwargs
import numpy as np

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import yaml
import time
import psutil, pickle, dill
import gym, ray, torch, argparse
from pydoc import locate
from ultra.env.agent_spec import UltraAgentSpec
from smarts.core.controllers import ActionSpaceType
from ultra.utils.episode import episodes
from ultra.src.evaluate import evaluation_check
import ultra_sac

num_gpus = 1 if torch.cuda.is_available() else 0


# @ray.remote(num_gpus=num_gpus / 2, max_calls=1)
@ray.remote(num_gpus=num_gpus / 2)
def train(
    task, policy_class, num_episodes, eval_info, timestep_sec, headless, etag, seed
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False

    # --------------------------------------------------------
    # Initialize Agent and social_vehicle encoding method
    # -------------------------------------------------------
    AGENT_ID = "007"
    spec = UltraAgentSpec(
        action_type=ActionSpaceType.Continuous, policy_class=policy_class
    )
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: spec},
        scenario_info=task,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )

    agent = spec.build_agent()

    for episode in episodes(num_episodes, etag=etag):
        observations = env.reset()
        state = observations[AGENT_ID]["state"]
        dones, infos = {"__all__": False}, None
        episode.reset()
        experiment_dir = episode.experiment_dir

        # save entire spec [ policy_params, reward_adapter, observation_adapter]
        if not os.path.exists(f"{experiment_dir}/spec.pkl"):
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            with open(f"{experiment_dir}/spec.pkl", "wb") as spec_output:
                dill.dump(spec, spec_output, pickle.HIGHEST_PROTOCOL)

        while not dones["__all__"]:
            if episode.get_itr(AGENT_ID) >= 1000000:  # 1M observation break
                finished = True
                break
            evaluation_check(
                agent=agent, agent_id=AGENT_ID, episode=episode, **eval_info, **env.info
            )
            action = agent.act(state, explore=True)
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            next_state = observations[AGENT_ID]["state"]

            # retrieve some relavant information from reward processor
            observations[AGENT_ID]["ego"].update(rewards[AGENT_ID]["log"])
            loss_output = agent.step(
                state=state,
                action=action,
                reward=rewards[AGENT_ID]["reward"],
                next_state=next_state,
                done=dones[AGENT_ID],
                max_steps_reached=observations[AGENT_ID]["ego"][
                    "events"
                ].reached_max_episode_steps,
            )
            episode.record_step(
                agent_id=AGENT_ID,
                observations=observations,
                rewards=rewards,
                total_step=total_step,
                loss_output=loss_output,
            )
            total_step += 1
            state = next_state

        episode.record_episode()
        episode.record_tensorboard(agent_id=AGENT_ID)
        if finished:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2, 3]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Tasks available : [easy, medium, hard, no-traffic]",
        type=str,
        default="easy",
    )
    parser.add_argument(
        "--episodes", help="number of training episodes", type=int, default=1000000
    )
    parser.add_argument(
        "--timestep", help="environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--eval-episodes", help="number of evaluation episodes", type=int, default=200
    )
    parser.add_argument(
        "--eval-rate",
        help="evaluation rate based on number of observations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--policy",
        help="path to policy class",
        default="TD3",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="environment seed",
        default=2,
        type=int,
    )
    args = parser.parse_args()

    with open("ultra/src/config.yaml", "r") as policy_file:
        policies = yaml.safe_load(policy_file)["policies"]
        if args.policy in policies:
            policy_class = locate(policies[args.policy])
        else:
            policy_class = locate(args.policy)

    num_cpus = max(
        1, psutil.cpu_count(logical=False) - 1
    )  # remove `logical=False` to use all cpus
    ray_kwargs = default_ray_kwargs(num_cpus=num_cpus, num_gpus=num_gpus)
    ray.init(**ray_kwargs)
    try:
        ray.wait(
            [
                train.remote(
                    task=(args.task, args.level),
                    policy_class=policy_class,
                    num_episodes=int(args.episodes),
                    eval_info={
                        "eval_rate": float(args.eval_rate),
                        "eval_episodes": int(args.eval_episodes),
                        "policy_class": policy_class,
                    },
                    timestep_sec=float(args.timestep),
                    headless=args.headless,
                    etag=args.policy,
                    seed=args.seed,
                )
            ]
        )
    finally:
        time.sleep(1)
        ray.shutdown()
