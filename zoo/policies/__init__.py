import sys
import os
import importlib.util
from pathlib import Path
from typing import Any, Dict

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import make, register

from .keep_lane_agent import KeepLaneAgent
from .non_interactive_agent import NonInteractiveAgent
from .waypoint_tracking_agent import WaypointTrackingAgent

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

register(
    locator="waypoint-tracking-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Tracker, max_episode_steps=300),
        agent_builder=WaypointTrackingAgent,
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


from smarts.env.multi_scenario_env import resolve_agent_interface


def load_config(path):
    import yaml

    config = None
    if path.exists():
        with open(path, "r") as file:
            config = yaml.safe_load(file)
    return config


def competition_entry(**kwargs):
    policy_path = kwargs.get("policy_path", None)

    from .competition_agent import CompetitionAgent

    def env_wrapper(env):
        import gym

        # import policy.py module
        wrapper_path = str(os.path.join(policy_path, "policy.py"))
        wrapper_spec = importlib.util.spec_from_file_location(
            "competition_wrapper", wrapper_path
        )
        wrapper_module = importlib.util.module_from_spec(wrapper_spec)
        sys.modules["competition_wrapper"] = wrapper_module
        if wrapper_spec:
            wrapper_spec.loader.exec_module(wrapper_module)

        wrappers = wrapper_module.submitted_wrappers()
        env = gym.Wrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)

        # delete competition wrapper module
        sys.modules.pop("competition_wrapper")
        del wrapper_module

        return env

    config = load_config(Path(os.path.join(policy_path, "config.yaml")))

    spec = AgentSpec(
        interface=resolve_agent_interface(
            img_meters=int(config["img_meters"]),
            img_pixels=int(config["img_pixels"]),
            action_space="TargetPose",
        ),
        agent_params={
            "policy_path": policy_path,
        },
        adapt_env=env_wrapper,
        agent_builder=CompetitionAgent,
    )

    return spec


root_path = str(Path(__file__).absolute().parents[2])

register(
    "competition_agent-v0",
    entry_point=competition_entry,
    policy_path=os.path.join(root_path, "competition/track1/submission"),
)
