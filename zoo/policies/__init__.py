import importlib
from pathlib import Path

from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    DoneCriteria,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import make, register

from .chase_via_points_agent import ChaseViaPointsAgent
from .keep_lane_agent import KeepLaneAgent
from .non_interactive_agent import NonInteractiveAgent
from .waypoint_tracking_agent import WaypointTrackingAgent

register(
    locator="non-interactive-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoint_paths=True, action=ActionSpaceType.TargetPose
        ),
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

register(
    locator="chase-via-points-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.LaneWithContinuousSpeed,
            done_criteria=DoneCriteria(
                collision=True,
                off_road=True,
                off_route=False,
                on_shoulder=False,
                wrong_way=False,
                not_moving=False,
                agents_alive=None,
                interest=None,
            ),
            accelerometer=False,
            drivable_area_grid_map=False,
            lane_positions=False,
            lidar_point_cloud=False,
            max_episode_steps=None,
            neighborhood_vehicle_states=False,
            occupancy_grid_map=False,
            top_down_rgb=False,
            road_waypoints=False,
            waypoint_paths=Waypoints(lookahead=80),
            signals=False,
        ),
        agent_builder=ChaseViaPointsAgent,
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


def _verify_installation(pkg: str, module: str):
    try:
        lib = importlib.import_module(module, pkg)
    except (ModuleNotFoundError, ImportError):
        raise ModuleNotFoundError(
            "Zoo agent is not installed. "
            f"Install via `scl zoo install {str(Path(__file__).resolve().parent/pkg)}`."
        )

    return lib


def entry_point_iamp(**kwargs):
    pkg = "interaction_aware_motion_prediction"
    module = ".policy"
    lib = _verify_installation(pkg=pkg, module=module)

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
        ),
        agent_builder=lib.Policy,
    )


register(
    locator="interaction-aware-motion-prediction-agent-v0", entry_point=entry_point_iamp
)


def entry_point_casl(**kwargs):
    pkg = "control_and_supervised_learning"
    module = ".policy"
    lib = _verify_installation(pkg=pkg, module=module)

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
        ),
        agent_builder=lib.Policy,
    )


register(
    locator="control-and-supervised-learning-agent-v0",
    entry_point=entry_point_casl,
)


def entry_point_dsac(**kwargs):
    pkg = "discrete_soft_actor_critic"
    module = ".policy"
    lib = _verify_installation(pkg=pkg, module=module)

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
        ),
        agent_builder=lib.Policy,
    )


register(locator="discrete-soft-actor-critic-agent-v0", entry_point=entry_point_dsac)


def open_entrypoint(*, debug: bool = False, aggressiveness: int = 3) -> AgentSpec:
    try:
        open_agent = importlib.import_module("open_agent")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Ensure that the open-agent has been installed with `pip install open-agent"
        )
    return open_agent.entrypoint(debug=debug, aggressiveness=aggressiveness)


register(locator="open-agent-v0", entry_point=open_entrypoint)
