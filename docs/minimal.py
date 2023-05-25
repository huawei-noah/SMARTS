import gymnasium as gym
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType

agent_id = "Agent-007"
agent_interface = AgentInterface.from_type(AgentType.Laner)

env = gym.make(
    "smarts.env:hiway-v1",
    scenarios=["scenarios/sumo/loop"],
    agent_interfaces={agent_id: agent_interface},
)

agent = Agent.from_function(agent_function=lambda _: "keep_lane")
observations = env.reset()
done = False
while not done:
    action = agent.act(observations[agent_id])
    observations, _, terminated, truncated, _ = env.step({agent_id: action})
    done = terminated or truncated

env.close()
