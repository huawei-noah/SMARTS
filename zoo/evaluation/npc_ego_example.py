import argparse
import logging
from importlib import import_module

import gym

from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)


def import_agent_specs_from(scenario_path):
    scenario_path = scenario_path.replace("/", ".")
    ego_agent_module = import_module(f"{scenario_path}.ego_agent")
    return ego_agent_module.agent_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("npc-ego-example")
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--headless", help="run simulation in headless mode", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent_specs = import_agent_specs_from(args.scenarios[0])

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=args.scenarios,
        agent_specs=agent_specs,
        headless=args.headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        endless_traffic=False,
        seed=args.seed,
        envision_record_data_replay_path=f"./{args.scenarios[0]}/data_replay",
    )

    for episode in episodes(n=1):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }

            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)

    env.close()
