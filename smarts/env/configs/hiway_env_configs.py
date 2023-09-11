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
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, List, Optional

from smarts.core.agent_interface import AgentInterface
from smarts.env.configs.base_config import EnvironmentArguments
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions


class ScenarioOrder(IntEnum):
    """Determines the order in which scenarios are served over successive resets."""

    sequential = 0
    """Scenarios are served in order initially provided."""
    scrambled = 1
    """Scenarios are served in random order."""
    default = scrambled
    """The default behavior. Defaults to ``scrambled``."""


class EnvReturnMode(IntEnum):
    """Configuration to determine the interface type of the step function.

    This configures between the environment status return (i.e. reward means the environment reward) and the per-agent
    status return (i.e. rewards means reward per agent).
    """

    per_agent = auto()
    """Generate per-agent mode step returns in the form ``(rewards({id: float}), terminateds({id: bool}), truncateds ({id: bool}), info)``."""
    environment = auto()
    """Generate environment mode step returns in the form ``(reward (float), terminated (bool), truncated (bool), info)``."""
    default = per_agent
    """The default behavior. Defaults to ``per_agent``."""


@dataclass
class SumoOptions:
    """Contains options used to configure sumo."""

    num_external_clients: int = 0
    """Number of SUMO clients beyond SMARTS. Defaults to 0."""
    auto_start: bool = True
    """Automatic starting of SUMO. Defaults to True."""
    headless: bool = True
    """If True, disables visualization in SUMO GUI. Defaults to True."""
    port: Optional[str] = None
    """SUMO port. Defaults to ``None``."""


@dataclass
class HiWayEnvV1Configuration(EnvironmentArguments):
    """The base configurations that should be used for HiWayEnvV1."""

    scenarios: List[str]
    """A list of scenario directories that will be simulated."""
    agent_interfaces: Dict[str, AgentInterface]
    """Specification of the agents needs that will be used to configure 
    the environment."""
    sim_name: Optional[str] = None
    """The name of the simulation."""
    scenarios_order: ScenarioOrder = ScenarioOrder.default
    """The order in which scenarios should be executed."""
    headless: bool = False
    """If this environment should attempt to connect to envision."""
    visdom: bool = False
    """Deprecated. Use SMARTS_VISDOM_ENABLED."""
    fixed_timestep_sec: float = 0.1
    """The time length of each step."""
    sumo_options: SumoOptions = field(default_factory=lambda: SumoOptions())
    """The configuration for the sumo instance."""
    seed: int = 42
    """The environment seed."""
    observation_options: ObservationOptions = ObservationOptions.default
    """Defines the options for how the formatting matches the observation space."""
    action_options: ActionOptions = ActionOptions.default
    """Defines the options for how the formatting matches the action space."""
    environment_return_mode: EnvReturnMode = EnvReturnMode.default
    """This configures between the environment step return information"""
