from dataclasses import fields
from typing import Callable, TypeVar, Union

from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts

T = TypeVar("T", Costs, Counts)


def add_dataclass(first: T, second: T) -> T:
    assert type(first) is type(second)
    new = {}
    for field in fields(first):
        new[field.name] = getattr(first, field.name) + getattr(second, field.name)
    output = first.__class__(**new)

    return output


def op_dataclass(
    first: T,
    second: Union[int, float],
    op: Callable[[Union[int, float], Union[int, float]], float],
) -> T:
    new = {}
    for field in fields(first):
        new[field.name] = op(getattr(first, field.name), second)
    output = first.__class__(**new)

    return output


def divide(value: Union[int, float], divider: Union[int, float]) -> float:
    return float(value / divider)


def multiply(value: Union[int, float], divider: Union[int, float]) -> float:
    return float(value * divider)
