from pathlib import Path
import importlib.resources as pkg_resources

from smarts.core.agent import AgentSpec
from .lane_space import (
    ACTION_SPACE,
    OBSERVATION_SPACE,
    get_action_adapter,
    agent_interface,
    get_observation_adapter,
)
from .policy import RLPolicy
from . import checkpoint
from smarts.zoo.registry import register


VERSION = 0.1


def entrypoint(
    goal_is_nearby_threshold=40,
    lane_end_threshold=51,
    lane_crash_distance_threshold=6,
    lane_crash_ttc_threshold=2,
    intersection_crash_distance_threshold=6,
    intersection_crash_ttc_threshold=5,
    target_speed=15,
    lane_change_speed=12.5,
):
    with pkg_resources.path(checkpoint, "checkpoint") as checkpoint_path:
        return AgentSpec(
            interface=agent_interface,
            observation_adapter=get_observation_adapter(
                goal_is_nearby_threshold=goal_is_nearby_threshold,
                lane_end_threshold=lane_end_threshold,
                lane_crash_distance_threshold=lane_crash_distance_threshold,
                lane_crash_ttc_threshold=lane_crash_ttc_threshold,
                intersection_crash_distance_threshold=intersection_crash_distance_threshold,
                intersection_crash_ttc_threshold=intersection_crash_ttc_threshold,
            ),
            action_adapter=get_action_adapter(
                target_speed=target_speed, lane_change_speed=lane_change_speed,
            ),
            policy_builder=lambda: RLPolicy(
                load_path=str(checkpoint_path.absolute()),
                policy_name="default_policy",
                observation_space=OBSERVATION_SPACE,
                action_space=ACTION_SPACE,
            ),
        )


register(locator="rl-agent-v0", entry_point=entrypoint)
