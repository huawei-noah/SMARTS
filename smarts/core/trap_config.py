from dataclasses import dataclass, field
from typing import Tuple, Union

from smarts.core.mission_planner import LapMission, Mission
from smarts.sstudio import types as sstudio_types


@dataclass(frozen=True)
class TrapConfig:
    zone: sstudio_types.Zone
    mission: Union[Mission, LapMission]
    reactivation_time: float
    activation_delay: float
    patience: float
    exclusion_prefixes: Tuple[str, ...] = field(default_factory=tuple)
