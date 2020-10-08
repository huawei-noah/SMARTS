from smarts.core.utils.class_factory import ClassRegister

agent_registry = ClassRegister()


def register(locator: str, entry_point, **kwargs):
    """Register an AgentSpec with the zoo.

    In order to load a registered AgentSpec it needs to be reachable from a
    directory contained in the PYTHONPATH.

    Args:
        locator:
            A string in the format of 'locator-name'
        entry_point:
            A callable that returns an AgentSpec or an AgentSpec object

    For example:

    .. code-block:: python

        register(
            locator="motion-planner-agent-v0",
            entry_point=lambda **kwargs: AgentSpec(
                interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
                policy_builder=MotionPlannerPolicy,
            ),
        )
    """

    agent_registry.register(locator=locator, entry_point=entry_point, **kwargs)


def make(locator: str, **kwargs):
    """Create an AgentSpec from the given locator.

    In order to load a registered AgentSpec it needs to be reachable from a
    directory contained in the PYTHONPATH.

    Args:
        locator:
            A string in the format of 'path.to.file:locator-name' where the path
            is in the form `{PYTHONPATH}[n]/path/to/file.py`
        kwargs:
            Additional arguments to be passed to the constructed class.
    """

    from smarts.core.agent import AgentSpec

    agent_spec = agent_registry.make(locator, **kwargs)
    assert isinstance(
        agent_spec, AgentSpec
    ), f"Expected make to produce an instance of AgentSpec, got: {agent_spec}"

    return agent_spec
