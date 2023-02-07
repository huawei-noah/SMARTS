import sys
from pathlib import Path
from typing import Any, Dict, Union

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.sstudio.scenario_construction import build_scenarios


class LaneFollowerAgent(Agent):
    def act(self, obs: Dict[Any, Union[Any, Dict]]):
        return (obs["waypoint_paths"]["speed_limit"][0][0], 0)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
    )

    env = HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={"SingleAgent": agent_interface},
        headless=headless,
    )

    for episode in episodes(n=num_episodes):
        agent = LaneFollowerAgent()
        observation, info = env.reset()
        episode.record_scenario(env.scenario_log)

        terminated = {"__all__": False}
        while not terminated["__all__"]:
            agent_action = agent.act(observation["SingleAgent"])
            observation, reward, terminated, truncated, info = env.step(
                {"SingleAgent": agent_action}
            )
            episode.record_step(observation, reward, terminated, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
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
