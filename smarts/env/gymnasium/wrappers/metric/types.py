from dataclasses import dataclass

from smarts.env.gymnasium.wrappers.metric.completion import (
    Completion,
    CompletionFuncs,
)
from smarts.env.gymnasium.wrappers.metric.costs import CostFuncs, Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts


@dataclass
class Record:
    """Stores an agent's scenario-completion, performance-count, and
    performance-cost values."""

    completion: Completion
    costs: Costs
    counts: Counts


@dataclass
class Data:
    """Stores an agent's performance-record, completion-functions, and
    cost-functions."""

    record: Record
    completion_funcs: CompletionFuncs
    cost_funcs: CostFuncs
