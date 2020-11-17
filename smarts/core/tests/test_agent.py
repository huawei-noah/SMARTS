from smarts.core.agent import AgentSpec, Agent


def test_building_agent_with_list_or_tuple_params():
    agent_spec = AgentSpec(
        agent_params=[32, 41],
        agent_builder=lambda x, y: Agent.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_tuple_params():
    agent_spec = AgentSpec(
        agent_params=(32, 41),
        agent_builder=lambda x, y: Agent.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_dict_params():
    agent_spec = AgentSpec(
        agent_params={"y": 2, "x": 1},
        agent_builder=lambda x, y: Agent.from_function(lambda _: x / y),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == 1 / 2
