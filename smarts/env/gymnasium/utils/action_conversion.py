import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
from cached_property import cached_property

from smarts.core.agent_interface import ActionSpaceType, AgentInterface

LINEAR_ACCELERATION_MINIMUM = -1e10
LINEAR_ACCELERATION_MAXIMUM = 1e10
ANGULAR_VELOCITY_MINIMUM = -1e10
ANGULAR_VELOCITY_MAXIMUM = 1e10
SPEED_MINIMUM = -1e10
SPEED_MAXIMUM = 1e10
POSITION_MINIMUM = 1e10
POSITION_MAXIMUM = 1e10
DT_MINIMUM = 1e-10
DT_MAXIMUM = 60.0

TRAJECTORY_LENGTH = 20
MPC_ARRAY_COUNT = 4


def _DEFAULT_PASSTHROUGH(action):
    return action


_throttle_break_steering_space = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

_actuator_dynamic_space = _throttle_break_steering_space


_continuous_space = _throttle_break_steering_space


_direct_space = gym.spaces.Box(
    low=np.array([LINEAR_ACCELERATION_MINIMUM, ANGULAR_VELOCITY_MINIMUM]),
    high=np.array([LINEAR_ACCELERATION_MAXIMUM, ANGULAR_VELOCITY_MAXIMUM]),
    dtype=np.float32,
)


_lane_space = gym.spaces.Discrete(n=4)


def _format_lane_space(action: int):
    _action_to_str = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    return _action_to_str[action]


_lane_with_continuous_speed_space = gym.spaces.Tuple(
    spaces=[
        gym.spaces.Box(
            low=SPEED_MINIMUM, high=SPEED_MAXIMUM, shape=(), dtype=np.float32
        ),
        gym.spaces.Box(low=100, high=100, shape=(), dtype=np.int8),
    ]
)

_base_trajectory_space = gym.spaces.Tuple(
    gym.spaces.Box(
        low=np.array([POSITION_MINIMUM] * TRAJECTORY_LENGTH),
        high=np.array([POSITION_MAXIMUM] * TRAJECTORY_LENGTH),
        dtype=np.float64,
    )
    for _ in range(MPC_ARRAY_COUNT)
)
_mpc_space = _base_trajectory_space

_base_target_pose_space = gym.spaces.Box(
    low=np.array([POSITION_MINIMUM, POSITION_MINIMUM, -math.pi, DT_MINIMUM]),
    high=np.array([POSITION_MAXIMUM, POSITION_MAXIMUM, math.pi, DT_MAXIMUM]),
    dtype=np.float64,
)
_multi_target_pose_space = _base_target_pose_space

_target_pose_space = _base_target_pose_space

_relative_target_pose_space = gym.spaces.Box(
    low=np.array([POSITION_MINIMUM, POSITION_MINIMUM, -math.pi]),
    high=np.array([POSITION_MAXIMUM, POSITION_MAXIMUM, math.pi]),
    dtype=np.float64,
)

_trajectory_space = _base_trajectory_space

_trajectory_with_time_space = gym.spaces.Tuple(
    [
        gym.spaces.Box(
            low=np.array([POSITION_MINIMUM] * TRAJECTORY_LENGTH),
            high=np.array([POSITION_MAXIMUM] * TRAJECTORY_LENGTH),
            dtype=np.float64,
        )
        for _ in range(MPC_ARRAY_COUNT)
    ]
    + [
        gym.spaces.Box(
            low=np.array([DT_MINIMUM] * TRAJECTORY_LENGTH),
            high=np.array([DT_MAXIMUM] * TRAJECTORY_LENGTH),
            dtype=np.float64,
        )
    ]
)


@dataclass()
class _FormattingGroup:
    space: gym.Space
    formatting_func: Callable[[Any], Any] = field(default=_DEFAULT_PASSTHROUGH)


@lru_cache(maxsize=1)
def get_formats() -> Dict[ActionSpaceType, _FormattingGroup]:
    return {
        ActionSpaceType.ActuatorDynamic: _FormattingGroup(
            space=_actuator_dynamic_space,
        ),
        ActionSpaceType.Continuous: _FormattingGroup(
            space=_continuous_space,
        ),
        ActionSpaceType.Direct: _FormattingGroup(
            space=_direct_space,
        ),
        ActionSpaceType.Empty: _FormattingGroup(
            space=gym.spaces.Tuple(spaces=()),
            formatting_func=lambda a: None,
        ),
        ActionSpaceType.Lane: _FormattingGroup(
            space=_lane_space,
            formatting_func=_format_lane_space,
        ),
        ActionSpaceType.LaneWithContinuousSpeed: _FormattingGroup(
            space=_lane_with_continuous_speed_space,
        ),
        ActionSpaceType.MPC: _FormattingGroup(
            space=_mpc_space,
        ),
        ActionSpaceType.MultiTargetPose: _FormattingGroup(
            space=_multi_target_pose_space,
        ),
        ActionSpaceType.RelativeTargetPose: _FormattingGroup(
            space=_relative_target_pose_space,
        ),
        ActionSpaceType.TargetPose: _FormattingGroup(
            space=_target_pose_space,
        ),
        ActionSpaceType.Trajectory: _FormattingGroup(
            space=_trajectory_space,
        ),
        ActionSpaceType.TrajectoryWithTime: _FormattingGroup(
            space=_trajectory_with_time_space,
        ),
    }


class ActionsSpaceFormatter:
    def __init__(self, agent_interfaces: Dict[str, AgentInterface]) -> None:
        self._agent_interfaces = agent_interfaces

    def format(self, actions: Dict[str, Any]):
        out_actions = {}
        for agent_id, action in actions.items():
            agent_interface = self._agent_interfaces[agent_id]
            format_ = get_formats()[agent_interface]
            assert format_.space.contains(
                action
            ), f"Action {action} does not match space {format_.space}!"

        return out_actions

    @cached_property
    def space(self):
        return gym.spaces.Dict(
            {
                agent_id: get_formats()[agent_interface.action].space
                for agent_id, agent_interface in self._agent_interfaces.items()
            }
        )
