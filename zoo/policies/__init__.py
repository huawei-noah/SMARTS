from typing import Any, Dict

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import make, register

from .keep_lane_agent import KeepLaneAgent
from .non_interactive_agent import NonInteractiveAgent

register(
    locator="non-interactive-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        agent_builder=NonInteractiveAgent,
        agent_params=kwargs,
    ),
)

register(
    locator="keep-lane-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        agent_builder=KeepLaneAgent,
    ),
)


def klws_entrypoint(speed):
    from .keep_left_with_speed_agent import KeepLeftWithSpeedAgent

    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=20000
        ),
        agent_params={"speed": speed * 0.01},
        agent_builder=KeepLeftWithSpeedAgent,
    )


register(locator="keep-left-with-speed-agent-v0", entry_point=klws_entrypoint)

social_index = 0
replay_save_dir = "./replay"
replay_read = False


def replay_entrypoint(
    save_directory,
    id,
    wrapped_agent_locator,
    wrapped_agent_params=None,
    read=False,
):
    if wrapped_agent_params is None:
        wrapped_agent_params = {}
    from .replay_agent import ReplayAgent

    internal_spec = make(wrapped_agent_locator, **wrapped_agent_params)
    global social_index
    global replay_save_dir
    global replay_read
    spec = AgentSpec(
        interface=internal_spec.interface,
        agent_params={
            "save_directory": replay_save_dir,
            "id": f"{id}_{social_index}",
            "internal_spec": internal_spec,
            "wrapped_agent_params": wrapped_agent_params,
            "read": replay_read,
        },
        agent_builder=ReplayAgent,
    )
    social_index += 1
    return spec


register(locator="replay-agent-v0", entry_point=replay_entrypoint)


def human_keyboard_entrypoint(*arg, **kwargs):
    from .human_in_the_loop import HumanKeyboardAgent

    spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering, max_episode_steps=3000
        ),
        agent_builder=HumanKeyboardAgent,
    )
    return spec


register(locator="human-in-the-loop-v0", entry_point=human_keyboard_entrypoint)
