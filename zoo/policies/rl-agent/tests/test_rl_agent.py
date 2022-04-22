from rl_agent import entrypoint


def test_building():
    agent_spec = entrypoint()
    agent = agent_spec.build_agent()
    assert hasattr(agent, "act")


def test_build_multiple_agents():
    agent_spec = entrypoint()
    agents = [agent_spec.build_agent() for _ in range(3)]

    assert all([hasattr(a, "act") for a in agents])
