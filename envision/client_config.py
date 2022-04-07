from dataclasses import dataclass
from typing import Any, Dict, NamedTuple


class ToggleOverride(NamedTuple):
    enabled: bool
    default: Any
    use_default = True


@dataclass(frozen=True)
class EnvisionStateFilter:
    actor_data_filter: Dict[str, ToggleOverride]
    simulation_data_filter: Dict[str, ToggleOverride]
    max_driven_path: int = 20
