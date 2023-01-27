# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
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
POSITION_COORDINATE_MINIMUM = -1e10
POSITION_COORDINATE_MAXIMUM = 1e10
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
        gym.spaces.Box(low=-100, high=100, shape=(), dtype=np.int8),
    ]
)

_base_trajectory_space = gym.spaces.Tuple(
    gym.spaces.Box(
        low=np.array([POSITION_COORDINATE_MINIMUM] * TRAJECTORY_LENGTH),
        high=np.array([POSITION_COORDINATE_MAXIMUM] * TRAJECTORY_LENGTH),
        dtype=np.float64,
    )
    for _ in range(MPC_ARRAY_COUNT)
)
_mpc_space = _base_trajectory_space

_base_target_pose_space = gym.spaces.Box(
    low=np.array(
        [POSITION_COORDINATE_MINIMUM, POSITION_COORDINATE_MINIMUM, -math.pi, DT_MINIMUM]
    ),
    high=np.array(
        [POSITION_COORDINATE_MAXIMUM, POSITION_COORDINATE_MAXIMUM, math.pi, DT_MAXIMUM]
    ),
    dtype=np.float64,
)
_multi_target_pose_space = _base_target_pose_space

_target_pose_space = _base_target_pose_space

_relative_target_pose_space = gym.spaces.Box(
    low=np.array([POSITION_COORDINATE_MINIMUM, POSITION_COORDINATE_MINIMUM, -math.pi]),
    high=np.array([POSITION_COORDINATE_MAXIMUM, POSITION_COORDINATE_MAXIMUM, math.pi]),
    dtype=np.float64,
)

_trajectory_space = _base_trajectory_space

_trajectory_with_time_space = gym.spaces.Tuple(
    [
        gym.spaces.Box(
            low=np.array([POSITION_COORDINATE_MINIMUM] * TRAJECTORY_LENGTH),
            high=np.array([POSITION_COORDINATE_MAXIMUM] * TRAJECTORY_LENGTH),
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


@dataclass(frozen=True)
class _FormattingGroup:
    space: gym.Space
    formatting_func: Callable[[Any], Any] = field(default=_DEFAULT_PASSTHROUGH)


@lru_cache(maxsize=1)
def get_formats() -> Dict[ActionSpaceType, _FormattingGroup]:
    """Get the currently available formatting groups for converting actions from `gym` space
    standard to SMARTS accepted observations.

    Returns:
        Dict[ActionSpaceType, _FormattingGroup]: The currently available formatting groups.
    """
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


class ActionOptions(IntEnum):
    """Defines the options for how the formatting matches the action space."""

    multi_agent = 0
    """Action must map to partial action space. Only active agents are included."""
    full = 1
    """Action must map to full action space. Inactive and active agents are included."""
    unformatted = 2
    """Actions are not reformatted or constrained to action space. Actions must directly map to
    underlying SMARTS actions."""
    default = 0
    """Defaults to :attr:`multi_agent`."""


class ActionSpacesFormatter:
    """Formats actions to adapt SMARTS to `gym` environment requirements.

    Args:
        agent_interfaces (Dict[str, AgentInterface]): The agent interfaces needed to determine the
            shape of the actions.
        action_options (ActionOptions): Options to configure the end formatting of the actions.
    """

    def __init__(
        self, agent_interfaces: Dict[str, AgentInterface], action_options: ActionOptions
    ) -> None:
        self._agent_interfaces = agent_interfaces
        self._action_options = action_options

        for agent_id, agent_interface in agent_interfaces.items():
            assert self.supported(agent_interface.action), (
                f"Agent `{agent_id}` is using an "
                f"unsupported `{agent_interface.action}`."
                f"Available actions:\n{json.dumps(set(agent_interfaces.keys()), indent=2)}"
            )

    def format(self, actions: Dict[str, Any]):
        """Format the action to a form that SMARTS can use.

        Args:
            actions (Dict[str, Any]): The actions to format.

        Returns:
            (Observation, Dict[str, Any]): The formatted actions.
        """

        if self._action_options == ActionOptions.unformatted:
            return actions

        out_actions = {}
        formatting_groups = get_formats()
        for agent_id, action in actions.items():
            agent_interface = self._agent_interfaces[agent_id]
            format_ = formatting_groups[agent_interface.action]
            space: gym.Space = self.space[agent_id]
            assert space is format_.space
            assert space.contains(
                action
            ), f"Action {action} does not match space {space}!"
            formatted_action = format_.formatting_func(action)
            out_actions[agent_id] = formatted_action

        if self._action_options == ActionOptions.full:
            assert actions.keys() == self.space.spaces.keys()

        return out_actions

    @staticmethod
    def supported(action_type: ActionSpaceType):
        """Test if the action is in the supported int

        Args:
            action_type (ActionSpaceType): The action type to check.

        Returns:
            bool: If the action type is supported by the formatter.
        """
        return action_type in get_formats()

    @cached_property
    def space(self) -> gym.spaces.Dict:
        """The action space given the current configuration.

        Returns:
            gym.spaces.Dict: A description of the action space that this formatter requires.
        """
        return gym.spaces.Dict(
            {
                agent_id: get_formats()[agent_interface.action].space
                for agent_id, agent_interface in self._agent_interfaces.items()
            }
        )
