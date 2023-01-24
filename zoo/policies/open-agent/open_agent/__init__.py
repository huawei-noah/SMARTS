from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


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
        "rate": 1,
    },
    debug=False,
    aggressiveness=0,
    max_episode_steps=None,
):
    from .agent import OpEnAgent

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.Trajectory,
            waypoint_paths=True,
            neighborhood_vehicle_states=True,
            max_episode_steps=max_episode_steps,
        ),
        agent_params={
            "gains": gains,
            "debug": debug,
        },
        agent_builder=OpEnAgent,
    )


register(locator="open_agent-v0", entry_point=entrypoint)
