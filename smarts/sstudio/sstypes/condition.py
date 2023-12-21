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
import enum
import sys
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from functools import cached_property
from typing import Type

from smarts.core.condition_state import ConditionState


class ConditionOperator(IntEnum):
    """Represents logical operators between conditions."""

    CONJUNCTION = enum.auto()
    """Evaluate true if both operands are true, otherwise false."""

    DISJUNCTION = enum.auto()
    """Evaluate true if either operand is true, otherwise false."""

    IMPLICATION = enum.auto()
    """Evaluate true if either the first operand is false, or both operands are true, otherwise false."""

    ## This would be desirable but makes the implementation more difficult in comparison to a negated condition.
    # NEGATION=enum.auto()
    # """True if its operand is false, otherwise false."""


class ConditionRequires(IntFlag):
    """This bitfield lays out the required information that a condition needs in order to evaluate."""

    none = 0

    # MISSION CONSTANTS
    agent_id = enum.auto()
    mission = enum.auto()

    # SIMULATION STATE
    time = enum.auto()
    actor_ids = enum.auto()
    actor_states = enum.auto()
    road_map = enum.auto()
    simulation = enum.auto()

    # ACTOR STATE
    current_actor_state = enum.auto()
    current_actor_road_status = enum.auto()

    any_simulation_state = time | actor_ids | actor_states | simulation
    any_current_actor_state = mission | current_actor_state | current_actor_road_status
    any_mission_state = agent_id | mission


@dataclass(frozen=True)
class Condition:
    """This encompasses an expression to evaluate to a logical result."""

    def evaluate(self, **kwargs) -> ConditionState:
        """Used to evaluate if a condition is met.

        Returns:
            ConditionState: The evaluation result of the condition.
        """
        raise NotImplementedError()

    @property
    def requires(self) -> ConditionRequires:
        """Information that the condition requires to evaluate state.

        Returns:
            ConditionRequires: The types of information this condition needs in order to evaluate.
        """
        raise NotImplementedError()

    def negation(self) -> "NegatedCondition":
        """Negates this condition giving the opposite result on evaluation.

        .. spelling:word-list::

            ConditionState

        >>> condition_true = LiteralCondition(ConditionState.TRUE)
        >>> condition_true.evaluate()
        <ConditionState.TRUE: 4>
        >>> condition_false = condition_true.negation()
        >>> condition_false.evaluate()
        <ConditionState.FALSE: 0>

        Note\\: This erases temporal values EXPIRED and BEFORE.
        >>> condition_before = LiteralCondition(ConditionState.BEFORE)
        >>> condition_before.negation().negation().evaluate()
        <ConditionState.FALSE: 0>

        Returns:
            NegatedCondition: The wrapped condition.
        """
        return NegatedCondition(self)

    def conjunction(self, other: "Condition") -> "CompoundCondition":
        """Resolve conditions as A AND B.

        The bit AND operator has been overloaded to call this method.
        >>> condition = DependeeActorCondition("leader")
        >>> condition.evaluate(actor_ids={"leader"})
        <ConditionState.TRUE: 4>
        >>> conjunction = condition & LiteralCondition(ConditionState.FALSE)
        >>> conjunction.evaluate(actor_ids={"leader"})
        <ConditionState.FALSE: 0>

        Note that the resolution has the priority EXPIRED > BEFORE > FALSE > TRUE.
        >>> conjunction = LiteralCondition(ConditionState.TRUE) & LiteralCondition(ConditionState.BEFORE)
        >>> conjunction.evaluate()
        <ConditionState.BEFORE: 1>
        >>> (conjunction & LiteralCondition(ConditionState.EXPIRED)).evaluate()
        <ConditionState.EXPIRED: 2>

        Returns:
            CompoundCondition: A condition combining two conditions using an AND operation.
        """
        return CompoundCondition(self, other, operator=ConditionOperator.CONJUNCTION)

    def disjunction(self, other: "Condition") -> "CompoundCondition":
        """Resolve conditions as A OR B.

        The bit OR operator has been overloaded to call this method.
        >>> disjunction = LiteralCondition(ConditionState.TRUE) | LiteralCondition(ConditionState.BEFORE)
        >>> disjunction.evaluate()
        <ConditionState.TRUE: 4>

        Note that the resolution has the priority TRUE > BEFORE > FALSE > EXPIRED.
        >>> disjunction = LiteralCondition(ConditionState.FALSE) | LiteralCondition(ConditionState.EXPIRED)
        >>> disjunction.evaluate()
        <ConditionState.FALSE: 0>
        >>> (disjunction | LiteralCondition(ConditionState.BEFORE)).evaluate()
        <ConditionState.BEFORE: 1>
        """
        return CompoundCondition(self, other, operator=ConditionOperator.DISJUNCTION)

    def implication(self, other: "Condition") -> "CompoundCondition":
        """Resolve conditions as A IMPLIES B. This is the same as A AND B OR NOT A."""
        return CompoundCondition(self, other, operator=ConditionOperator.IMPLICATION)

    def trigger(
        self, delay_seconds: float, persistent: bool = False
    ) -> "ConditionTrigger":
        """Converts the condition to a trigger which becomes permanently TRUE after the first time the inner condition becomes TRUE.

        >>> trigger = TimeWindowCondition(2, 5).trigger(delay_seconds=0)
        >>> trigger.evaluate(time=1)
        <ConditionState.BEFORE: 1>
        >>> trigger.evaluate(time=4)
        <ConditionState.TRUE: 4>
        >>> trigger.evaluate(time=90)
        <ConditionState.TRUE: 4>

        >>> start_time = 5
        >>> between_time = 10
        >>> delay_seconds = 20
        >>> trigger = LiteralCondition(ConditionState.TRUE).trigger(delay_seconds=delay_seconds)
        >>> trigger.evaluate(time=start_time)
        <ConditionState.BEFORE: 1>
        >>> trigger.evaluate(time=between_time)
        <ConditionState.BEFORE: 1>
        >>> trigger.evaluate(time=start_time + delay_seconds)
        <ConditionState.TRUE: 4>
        >>> trigger.evaluate(time=between_time)
        <ConditionState.BEFORE: 1>

        Args:
            delay_seconds (float): Applies the trigger after the delay has passed since the inner condition first TRUE. Defaults to False.
            persistent (bool, optional): Mixes the inner result with the trigger result using an AND operation.

        Returns:
            ConditionTrigger: A resulting condition.
        """
        return ConditionTrigger(
            self, delay_seconds=delay_seconds, persistent=persistent
        )

    def expire(
        self, time, expired_state=ConditionState.EXPIRED, relative: bool = False
    ) -> "ExpireTrigger":
        """This trigger evaluates to the expired state value after the given simulation time.

        >>> trigger = LiteralCondition(ConditionState.TRUE).expire(20)
        >>> trigger.evaluate(time=10)
        <ConditionState.TRUE: 4>
        >>> trigger.evaluate(time=30)
        <ConditionState.EXPIRED: 2>

        Args:
            time (float): The simulation time when this trigger changes.
            expired_state (ConditionState, optional): The condition state to use when the simulation is after the given time. Defaults to ConditionState.EXPIRED.
            relative (bool, optional): If this trigger should resolve relative to the first evaluated time.
        Returns:
            ExpireTrigger: The resulting condition.
        """
        return ExpireTrigger(
            inner_condition=self,
            time=time,
            expired_state=expired_state,
            relative=relative,
        )

    def __and__(self, other: "Condition") -> "CompoundCondition":
        """Resolve conditions as A AND B."""
        assert isinstance(other, Condition)
        return self.conjunction(other)

    def __or__(self, other: "Condition") -> "CompoundCondition":
        """Resolve conditions as A OR B."""
        assert isinstance(other, Condition)
        return self.disjunction(other)

    def __neg__(self) -> "NegatedCondition":
        """Negates this condition"""
        return self.negation()


@dataclass(frozen=True)
class SubjectCondition(Condition):
    """This condition assumes that there is a subject involved."""

    def evaluate(self, **kwargs) -> ConditionState:
        """Used to evaluate if a condition is met.

        Args:
            actor_info: Information about the currently relevant actor.
        Returns:
            ConditionState: The evaluation result of the condition.
        """
        raise NotImplementedError()

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.current_actor_state


_abstract_conditions = (Condition, SubjectCondition)


@dataclass(frozen=True)
class LiteralCondition(Condition):
    """This condition evaluates as a literal without considering evaluation parameters."""

    literal: ConditionState
    """The literal value of this condition."""

    def evaluate(self, **kwargs) -> ConditionState:
        return self.literal

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.none


@dataclass(frozen=True)
class TimeWindowCondition(Condition):
    """This condition should be true in the given simulation time window."""

    start: float
    """The starting simulation time before which this condition becomes false."""
    end: float
    """The ending simulation time as of which this condition becomes expired."""

    def evaluate(self, **kwargs) -> ConditionState:
        time = kwargs[ConditionRequires.time.name]
        if self.start <= time < self.end or self.end == sys.maxsize:
            return ConditionState.TRUE
        elif time > self.end:
            return ConditionState.EXPIRED
        return ConditionState.BEFORE

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.time


@dataclass(frozen=True)
class DependeeActorCondition(Condition):
    """This condition should be true if the given actor exists."""

    actor_id: str
    """The id of an actor in the simulation that needs to exist for this condition to be true."""

    def evaluate(self, **kwargs) -> ConditionState:
        actor_ids = kwargs[self.requires.name]
        if self.actor_id in actor_ids:
            return ConditionState.TRUE
        return ConditionState.FALSE

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.actor_ids

    def __post_init__(self):
        assert isinstance(self.actor_id, str)


@dataclass(frozen=True)
class NegatedCondition(Condition):
    """This condition negates the inner condition to flip between TRUE and FALSE.

    Note\\: This erases temporal values EXPIRED and BEFORE.
    """

    inner_condition: Condition
    """The inner condition to negate."""

    def evaluate(self, **kwargs) -> ConditionState:
        result = self.inner_condition.evaluate(**kwargs)
        if ConditionState.TRUE in result:
            return ConditionState.FALSE
        return ConditionState.TRUE

    @property
    def requires(self) -> ConditionRequires:
        return self.inner_condition.requires

    def __post_init__(self):
        if self.inner_condition.__class__ in _abstract_conditions:
            raise TypeError(
                f"Abstract `{self.inner_condition.__class__.__name__}` cannot use the negation operation."
            )


@dataclass(frozen=True)
class ExpireTrigger(Condition):
    """This condition allows for expiration after a given time."""

    inner_condition: Condition
    """The inner condition to delay."""

    time: float
    """The simulation time when this trigger becomes expired."""

    expired_state: ConditionState = ConditionState.EXPIRED
    """The state value this trigger should have when it expires."""

    relative: bool = False
    """If this should start relative to the first time evaluated."""

    def evaluate(self, **kwargs) -> ConditionState:
        time = kwargs[ConditionRequires.time.name]
        if self.relative:
            key = "met"
            met_time = getattr(self, key, -1)
            if met_time == -1:
                object.__setattr__(self, key, time)
                time = 0
            else:
                time -= met_time
        if time >= self.time:
            return self.expired_state
        return self.inner_condition.evaluate(**kwargs)

    @cached_property
    def requires(self) -> ConditionRequires:
        return self.inner_condition.requires | ConditionRequires.time

    def __post_init__(self):
        if self.inner_condition.__class__ in _abstract_conditions:
            raise TypeError(
                f"Abstract `{self.inner_condition.__class__.__name__}` cannot be wrapped by a trigger."
            )


@dataclass(frozen=True)
class ConditionTrigger(Condition):
    """This condition is a trigger that assumes an untriggered constant state and then turns to the other state permanently
    on the inner condition becoming TRUE. There is also an option to delay response to the the inner condition by a number
    of seconds. This will convey an EXPIRED value immediately because that state means the inner value will never be TRUE.

    This can be used to wait for some time after the inner condition has become TRUE to trigger.
    Note that the original condition may no longer be true by the time delay has expired.

    This will never resolve TRUE on the first evaluate.
    """

    inner_condition: Condition
    """The inner condition to delay."""

    delay_seconds: float
    """The number of seconds to delay for."""

    untriggered_state: ConditionState = ConditionState.BEFORE
    """The state before the inner trigger condition and delay is resolved."""

    triggered_state: ConditionState = ConditionState.TRUE
    """The state after the inner trigger condition and delay is resolved."""

    persistent: bool = False
    """If the inner condition state is used in conjunction with the triggered state. (inner_condition_state & triggered_state)"""

    def evaluate(self, **kwargs) -> ConditionState:
        time = kwargs[ConditionRequires.time.name]
        key = "met_time"
        result = self.untriggered_state
        met_time = getattr(self, key, -1)
        if met_time == -1:
            if self.inner_condition.evaluate(**kwargs):
                object.__setattr__(self, key, time)
                time = 0
            else:
                time = -1
        else:
            time -= met_time
        if time >= self.delay_seconds:
            result = self.triggered_state
            if self.persistent:
                result &= self.inner_condition.evaluate(**kwargs)
            return result

        temporals = result & (ConditionState.EXPIRED)
        if ConditionState.EXPIRED in temporals:
            return ConditionState.EXPIRED
        return self.untriggered_state

    @property
    def requires(self) -> ConditionRequires:
        return self.inner_condition.requires | ConditionRequires.time

    def __post_init__(self):
        if self.inner_condition.__class__ in _abstract_conditions:
            raise TypeError(
                f"Abstract `{self.inner_condition.__class__.__name__}` cannot be wrapped by a trigger."
            )
        if self.delay_seconds < 0:
            raise ValueError("Delay cannot be negative.")


@dataclass(frozen=True)
class OffRoadCondition(SubjectCondition):
    """This condition is true if the subject is on road."""

    def evaluate(self, **kwargs) -> ConditionState:
        current_actor_road_status = kwargs[self.requires.name]
        if (
            current_actor_road_status.road is None
            and not current_actor_road_status.off_road
        ):
            return ConditionState.BEFORE
        return (
            ConditionState.TRUE
            if current_actor_road_status.off_road
            else ConditionState.FALSE
        )

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.current_actor_road_status


@dataclass(frozen=True)
class VehicleTypeCondition(SubjectCondition):
    """This condition is true if the subject is of the given vehicle types."""

    vehicle_type: str

    def evaluate(self, **kwargs) -> ConditionState:
        current_actor_state = kwargs[self.requires.name]
        return (
            ConditionState.TRUE
            if current_actor_state.vehicle_config_type == self.vehicle_type
            else ConditionState.FALSE
        )

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.current_actor_state


@dataclass(frozen=True)
class VehicleSpeedCondition(SubjectCondition):
    """This condition is true if the subject has a speed between low and high."""

    low: float
    """The lowest speed allowed."""

    high: float
    """The highest speed allowed."""

    def evaluate(self, **kwargs) -> ConditionState:
        vehicle_state = kwargs[self.requires.name]
        return (
            ConditionState.TRUE
            if self.low <= vehicle_state.speed <= self.high
            else ConditionState.FALSE
        )

    @property
    def requires(self) -> ConditionRequires:
        return ConditionRequires.current_actor_state

    @classmethod
    def loitering(cls: Type["VehicleSpeedCondition"], abs_error=0.01):
        """Generates a speed condition which assumes that the subject is stationary."""
        return cls(low=abs_error, high=abs_error)


@dataclass(frozen=True)
class CompoundCondition(Condition):
    """This compounds multiple conditions.

    The following cases are notable
        CONJUNCTION (A AND B)
            If both conditions evaluate TRUE the result is exclusively TRUE.
            Else if either condition evaluates EXPIRED the result will be EXPIRED.
            Else if either condition evaluates BEFORE the result will be BEFORE.
            Else FALSE
        DISJUNCTION (A OR B)
            If either condition evaluates TRUE the result is exclusively TRUE.
            Else if either condition evaluates BEFORE then the result will be BEFORE.
            Else if both conditions evaluate EXPIRED then the result will be EXPIRED.
            Else FALSE
        IMPLICATION (A AND B or not A)
            If the first condition evaluates *not* TRUE the result is exclusively TRUE.
            Else if the first condition evaluates TRUE and the second condition evaluates TRUE the result is exclusively TRUE.
            Else FALSE
    """

    first_condition: Condition
    """The first condition."""

    second_condition: Condition
    """The second condition."""

    operator: ConditionOperator
    """The operator used to combine these conditions."""

    def evaluate(self, **kwargs) -> ConditionState:
        # Short circuits
        first_eval = self.first_condition.evaluate(**kwargs)
        if (
            self.operator == ConditionOperator.CONJUNCTION
            and ConditionState.EXPIRED in first_eval
        ):
            return ConditionState.EXPIRED
        elif (
            self.operator == ConditionOperator.DISJUNCTION
            and ConditionState.TRUE in first_eval
        ):
            return ConditionState.TRUE
        elif (
            self.operator == ConditionOperator.IMPLICATION
            and ConditionState.TRUE not in first_eval
        ):
            return ConditionState.TRUE

        second_eval = self.second_condition.evaluate(**kwargs)
        if (
            self.operator == ConditionOperator.IMPLICATION
            and ConditionState.TRUE in first_eval
            and ConditionState.TRUE in second_eval
        ):
            return ConditionState.TRUE

        elif self.operator == ConditionOperator.CONJUNCTION:
            conjunction = first_eval & second_eval
            if ConditionState.TRUE in conjunction:
                return ConditionState.TRUE

            # To priority of temporal versions of FALSE
            disjunction = first_eval | second_eval
            if ConditionState.EXPIRED in disjunction:
                return ConditionState.EXPIRED

            if ConditionState.BEFORE in disjunction:
                return ConditionState.BEFORE

        elif self.operator == ConditionOperator.DISJUNCTION:
            result = first_eval | second_eval

            if ConditionState.TRUE in result:
                return ConditionState.TRUE

            if ConditionState.BEFORE in result:
                return ConditionState.BEFORE

            if ConditionState.EXPIRED in first_eval & second_eval:
                return ConditionState.EXPIRED

        return ConditionState.FALSE

    @cached_property
    def requires(self) -> ConditionRequires:
        return self.first_condition.requires | self.second_condition.requires

    def __post_init__(self):
        for condition in (self.first_condition, self.second_condition):
            if condition.__class__ in _abstract_conditions:
                raise TypeError(
                    f"Abstract `{condition.__class__.__name__}` cannot use compound operations."
                )
