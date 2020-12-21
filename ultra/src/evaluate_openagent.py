import numpy as np
import gym, ray
import glob, os, datetime
import torch, argparse, random
from pydoc import locate
from ultra.utils.episode import Episode, LogInfo, episodes
from ultra.src.adapter import IntersectionAdapter
from smarts.core.agent import AgentSpec
import pdb, traceback, sys, yaml
import open_agent
from ultra.baselines.configs import Config

num_gpus = 1 if torch.cuda.is_available() else 0

# Number of GPUs should be splited between remote functions.
@ray.remote(num_gpus=num_gpus / 2)
def evaluate(
    seed, agent_id, checkpoint_dir, scenario_info, num_episodes, headless, timestep_sec
):
    open_agent_spec = open_agent.entrypoint(debug=False)
    agent = open_agent_spec.build_agent()
    agent.id = agent_id
    config = Config()
    # ultra_adapter = IntersectionAdapter(
    #     agent_id=agent_id,
    #     social_vehicle_config=config.social_vehicle_config,
    #     timestep_sec=timestep_sec,
    #     action_size=0,
    #     num_lookahead=5,
    # )

    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={agent_id: open_agent_spec},
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
        eval_mode=True,
    )

    summary_log = LogInfo()
    logs = []
    for episode in episodes(num_episodes):
        observations = env.reset()

        state = observations[agent_id]
        dones, infos = {"__all__": False}, None

        episode.reset(mode="Evaluation")
        while not dones["__all__"]:
            # ultra_obs = ultra_adapter.observation_adapter(observations[agent_id])
            # print('***', ultra_obs['ego'])
            action = agent.act(state)
            observations, rewards, dones, _ = env.step({agent_id: action})

            next_state = observations[agent_id]

            ultra_reward = calculate_my_reward(observations)

            ultra_obs["ego"].update(ultra_reward["log"])

            done = dones[agent_id]
            state = next_state
            episode.record_step(
                agent_id=agent_id,
                observations={agent_id: ultra_obs},
                rewards={agent_id: ultra_reward},
            )

        episode.record_episode()
        logs.append(episode.info[episode.active_tag].data)

        for key, value in episode.info[episode.active_tag].data.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                summary_log.data[key] += value

    for key, val in summary_log.data.items():
        if not isinstance(val, (list, tuple, np.ndarray)):
            summary_log.data[key] /= num_episodes

    for i in range(10000):
        episode.agents_itr[agent.id] += 1
        episode.record_tensorboard(agent_id=agent.id)
    env.close()

    return summary_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent-evaluation")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Tasks available : [easy, medium, hard]",
        type=str,
        default="easy",
    )

    parser.add_argument(
        "--episodes", help="number of training episodes", type=int, default=200
    )
    parser.add_argument(
        "--timestep", help="environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="run without envision", type=bool, default=False
    )
    args = parser.parse_args()

    ray.init()
    ray.wait(
        [
            evaluate.remote(
                agent_id="AGENT_008",
                seed=0,
                checkpoint_dir=None,
                scenario_info=(args.task, args.level),
                num_episodes=int(args.episodes),
                timestep_sec=float(args.timestep),
                headless=args.headless,
            )
        ]
    )
