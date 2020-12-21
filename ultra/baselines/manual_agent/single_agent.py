import logging
import argparse
import glob

import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

from smarts.core.agent import AgentSpec
from baselines.manual_agent.policy import ManualPolicy

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"

if __name__ == "__main__":

    parser = argparse.ArgumentParser("hiway-single-agent-example")
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
    args = parser.parse_args()

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Standard, max_episode_steps=3000),
        policy_builder=ManualPolicy,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=glob.glob(f"{args.scenarios[0]}/task1/train/2lane_t_40kmh_0/"),
        agent_specs={AGENT_ID: agent_spec},
        headless=args.headless,
        visdom=True,
        timestep_sec=0.1,
    )

    for episode in episodes(n=20):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()
