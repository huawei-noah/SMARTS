from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BehaviorMetric:
    safety: Any
    agility: Any
    stability: Any
    control_diversity: Any
    cut_in_ratio: Any


@dataclass
class GameTheoryMetric:
    collaborative: Any
    competitive: Any


@dataclass
class PerformanceMetric:
    collision_rate: float
    completion_rate: float
    generalization: Any
