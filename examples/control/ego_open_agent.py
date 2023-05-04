"""
This examples runs the OpEn Agent, a classical, MPC based agent using [OpEn](https://alphaville.github.io/optimization-engine/).
For further reading, see zoo policy [open-agent](../../smarts/zoo/policies/open-agent/).
"""

import importlib
import sys
from pathlib import Path

import gym

from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser

try:
    open_agent = importlib.import_module("open_agent")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Ensure that the open-agent has been installed with `pip install open-agent"
    )


AGENT_ID = "Agent-007"


def main(scenarios, headless, num_episodes):
    open_agent_spec = open_agent.entrypoint(debug=False, aggressiveness=3)
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: open_agent_spec.interface},
        headless=headless,
        sumo_headless=True,
    )

    for episode in episodes(n=num_episodes):
        agent = open_agent_spec.build_agent()

        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

        del agent

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("OpEn-trajectory-optimizer")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
