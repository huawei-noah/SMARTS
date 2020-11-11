from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register
from smarts.core.agent_interface import Task


def entrypoint(
    gains={
        "theta": 3.0,
        "position": 4.0,
        "obstacle": 3.0,
        "u_accel": 0.1,
        "u_yaw_rate": 1.0,
        "terminal": 0.01,
        "impatience": 0.01,
        "speed": 0.01,
    },
    debug=False,
    max_episode_steps=600,
):
    from .policy import Policy

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.Trajectory,
            waypoints=True,
            neighborhood_vehicles=True,
            max_episode_steps=max_episode_steps,
            task=Task.UTurn,
        ),
        policy_params={"gains": gains, "debug": debug,},
        policy_builder=Policy,
        perform_self_test=False,
    )


register(locator="open_agent-v0", entry_point=entrypoint)
