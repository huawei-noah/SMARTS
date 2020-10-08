from smarts.core.agent import AgentSpec, AgentPolicy


def test_building_agent_with_list_or_tuple_params():
    agent_spec = AgentSpec(
        policy_params=[32, 41],
        policy_builder=lambda x, y: AgentPolicy.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_tuple_params():
    agent_spec = AgentSpec(
        policy_params=(32, 41),
        policy_builder=lambda x, y: AgentPolicy.from_function(lambda _: (x, y)),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == (32, 41)


def test_building_agent_with_dict_params():
    agent_spec = AgentSpec(
        policy_params={"y": 2, "x": 1},
        policy_builder=lambda x, y: AgentPolicy.from_function(lambda _: x / y),
    )

    agent = agent_spec.build_agent()
    assert agent.act("dummy observation") == 1 / 2
