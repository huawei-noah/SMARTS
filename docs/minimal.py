import gym
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy

agent_id = "Agent-007"
agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner),
    policy_params={"policy_function": lambda _: "keep_lane"},
    policy_builder=AgentPolicy.from_function,
)

env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/loop"],
    agent_specs={agent_id: agent_spec},
)

agent = agent_spec.build_agent()
observations = env.reset()
dones = {"__all__": False}
while not dones["__all__"]:
    action = agent.act(observations[agent_id])
    observations, _, dones, _ = env.step({agent_id: action})

env.close()
