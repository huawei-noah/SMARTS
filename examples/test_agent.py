from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.utils.episodes import episodes

from examples.argument_parser import default_argument_parser

import gym
N_AGENTS = 1
AGENT_IDS = ["Agent %i" % i for i in range(N_AGENTS)]


class KeepLaneAgent(Agent):
    def act(self, obs):
        return (5, 0)
agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.LanerWithSpeed, max_episode_steps=500
            ),
            agent_builder=KeepLaneAgent,
        )

        for agent_id in AGENT_IDS
    }
from smarts.env.hiway_env import HiWayEnv
env = HiWayEnv(scenarios=['scenarios/intersections/2lane'],
        agent_specs=agent_specs,
        headless=False,
        seed=42,
    )
agents = {
    agent_id: agent_spec.build_agent()
    for agent_id, agent_spec in agent_specs.items()
}
observations = env.reset()

for _ in range(500):
    agent_actions = {
        agent_id: agents[agent_id].act(agent_obs)
        for agent_id, agent_obs in observations.items()
    }
    observations, _, _, _ = env.step(agent_actions)
    events = observations['Agent 0'].events
    if events.wrong_way:
        print(observations['Agent 0'].ego_vehicle_state.lane_id)
        break