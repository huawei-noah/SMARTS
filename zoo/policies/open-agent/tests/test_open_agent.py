from open_agent import entrypoint


def test_building():
    agent_spec = entrypoint()
    agent = agent_spec.build_agent()
    assert agent.policy.is_planner_running()


def test_build_multiple_agents():
    agent_spec = entrypoint()
    agents = [agent_spec.build_agent() for _ in range(3)]

    for agent in agents:
        assert agent.policy.is_planner_running()
