import sys
from pathlib import Path

import gym

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "Agent-007"


class TrackingAgent(Agent):
    def act(self, obs):
        lane_index = 0
        num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        # Desired speed is in m/s
        desired_speed = 50 / 3.6
        trajectory = [
            [
                obs.waypoint_paths[lane_index][i].pos[0]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].pos[1]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].heading
                for i in range(num_trajectory_points)
            ],
            [desired_speed for i in range(num_trajectory_points)],
        ]
        return trajectory


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Tracker, max_episode_steps=max_episode_steps
        ),
        agent_builder=TrackingAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        headless=headless,
        sumo_headless=True,
    )

    for episode in episodes(n=num_episodes):
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


if __name__ == "__main__":
    parser = default_argument_parser("trajectory-tracking")
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
