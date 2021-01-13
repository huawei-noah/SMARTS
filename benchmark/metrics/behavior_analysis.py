from typing import Dict, Any

from dataclasses import dataclass, field


@dataclass
class BehaviorMetric:
    safety: float = 0.0
    agility: float = 0.0
    stability: float = 0.0
    control_diversity: float = 0.0
    cut_in_ratio: float = 0.0

    def __post_init__(self):
        self._features = ["Safety", "Agility", "Stability", "Diversity"]

    @property
    def features(self):
        return self._features

    def compute(self, handler):
        raise NotImplementedError


@dataclass
class GameTheoryMetric:
    collaborative: Any
    competitive: Any


@dataclass
class PerformanceMetric:
    collision_rate: float
    completion_rate: float
    generalization: Any
