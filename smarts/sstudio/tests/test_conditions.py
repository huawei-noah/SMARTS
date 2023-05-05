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
from unittest.mock import MagicMock, Mock

import pytest

from smarts.sstudio.types import (
    CompoundCondition,
    Condition,
    ConditionOperator,
    ConditionState,
    DelayCondition,
    DependeeActorCondition,
    LiteralCondition,
    NegatedCondition,
    OnRoadCondition,
    SubjectCondition,
    TimeWindowCondition,
    VehicleTypeCondition,
)


def test_condition_state():
    assert not bool(ConditionState.BEFORE)
    assert not bool(ConditionState.EXPIRED)
    assert not bool(ConditionState.FALSE)
    assert bool(ConditionState.TRUE)

    assert ConditionState.TRUE
    assert bool(not ConditionState.TRUE) == False
    assert not ~ConditionState.TRUE

    assert bool(ConditionState.FALSE) == False
    assert not ConditionState.FALSE
    assert ~ConditionState.FALSE

    assert bool(ConditionState.EXPIRED) == False
    assert not ConditionState.EXPIRED
    assert ~ConditionState.EXPIRED

    assert bool(ConditionState.BEFORE) == False
    assert not ConditionState.BEFORE
    assert ~ConditionState.BEFORE

    assert ConditionState.TRUE | ConditionState.FALSE
    assert not ConditionState.TRUE & ConditionState.FALSE
    assert (
        ConditionState.TRUE
        | ConditionState.EXPIRED
        | ConditionState.FALSE
        | ConditionState.BEFORE
    )
    assert not (ConditionState.EXPIRED | ConditionState.FALSE | ConditionState.BEFORE)


def test_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)

    condition = Condition()

    with pytest.raises(NotImplementedError):
        condition.evaluate(actor_info=None)

    with pytest.raises(TypeError):
        condition.negate()

    with pytest.raises(TypeError):
        condition.conjoin(literal_true)

    with pytest.raises(TypeError):
        condition.disjoin(literal_true)

    with pytest.raises(TypeError):
        condition.implicate(literal_true)

    with pytest.raises(TypeError):
        condition.delay(10)


def test_compound_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)
    literal_false = LiteralCondition(ConditionState.FALSE)

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.CONJUNCTION,
    ) == literal_true.conjoin(literal_false)
    assert literal_true.conjoin(literal_true).evaluate()
    assert not literal_true.conjoin(literal_false).evaluate()
    assert not literal_false.conjoin(literal_true).evaluate()
    assert not literal_false.conjoin(literal_false).evaluate()

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.DISJUNCTION,
    ) == literal_true.disjoin(literal_false)
    assert literal_true.disjoin(literal_true)
    assert literal_true.disjoin(literal_false).evaluate()
    assert literal_false.disjoin(literal_true).evaluate()
    assert not literal_false.disjoin(literal_false).evaluate()

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.IMPLICATION,
    ) == literal_true.implicate(literal_false)
    assert literal_true.implicate(literal_true).evaluate()
    assert not literal_true.implicate(literal_false).evaluate()
    assert literal_false.implicate(literal_true).evaluate()
    assert literal_false.implicate(literal_false).evaluate()


def test_delay_condition():
    short_delay = 4
    long_delay = 10
    first_time_window_true = 5
    window_condition = TimeWindowCondition(4, 10)
    delayed_condition = window_condition.delay(long_delay, persistant=False)

    assert delayed_condition == DelayCondition(
        inner_condition=window_condition,
        seconds=long_delay,
        persistant=False,
    )

    # before
    time = 2
    assert (
        not delayed_condition.evaluate(simulation_time=time)
    ) and not window_condition.evaluate(simulation_time=time)
    # first true
    time = first_time_window_true
    assert (
        not delayed_condition.evaluate(simulation_time=time)
    ) and window_condition.evaluate(simulation_time=time)
    # delay not expired
    time = first_time_window_true + long_delay - 1
    assert (
        not delayed_condition.evaluate(simulation_time=time)
    ) and not window_condition.evaluate(simulation_time=time)
    # delay expired
    time = first_time_window_true + long_delay
    assert delayed_condition.evaluate(
        simulation_time=time
    ) and not window_condition.evaluate(simulation_time=time)
    # delay expired
    time = first_time_window_true + long_delay + 1
    assert delayed_condition.evaluate(
        simulation_time=time
    ) and not window_condition.evaluate(simulation_time=time)
    # delay not expired
    time = first_time_window_true + long_delay - 1
    assert not delayed_condition.evaluate(simulation_time=time)

    # Test persistant true
    delayed_condition = window_condition.delay(short_delay, persistant=True)
    time = first_time_window_true
    assert not delayed_condition.evaluate(simulation_time=time)
    time = first_time_window_true + short_delay
    assert delayed_condition.evaluate(simulation_time=time)
    time = first_time_window_true + long_delay
    assert not delayed_condition.evaluate(simulation_time=time)


def test_dependee_condition():
    dependee_condition = DependeeActorCondition("leader")
    pass


def test_literal_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)
    literal_false = LiteralCondition(ConditionState.FALSE)

    assert literal_false.evaluate() == ConditionState.FALSE
    assert literal_true.evaluate() == ConditionState.TRUE
    assert literal_true.evaluate()
    assert not literal_false.evaluate()


def test_negated_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)
    literal_false = LiteralCondition(ConditionState.FALSE)

    assert literal_false.negate() == NegatedCondition(literal_false)
    assert literal_true.negate() == NegatedCondition(literal_true)

    assert literal_false.negate().evaluate()
    assert not literal_true.negate().evaluate()


def test_on_road_condition():
    on_road_condition = OnRoadCondition()
    pass


def test_time_window_condition():
    start = 4
    between = 8
    end = 10

    window_condition = TimeWindowCondition(start=start, end=end)

    assert not window_condition.evaluate(simulation_time=start - 1)
    assert window_condition.evaluate(simulation_time=start)
    assert window_condition.evaluate(simulation_time=between)
    assert not window_condition.evaluate(simulation_time=end)


def test_subject_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)

    subject_condition = SubjectCondition()

    with pytest.raises(NotImplementedError):
        subject_condition.evaluate(actor_info=None)

    with pytest.raises(TypeError):
        subject_condition.negate()

    with pytest.raises(TypeError):
        subject_condition.conjoin(literal_true)

    with pytest.raises(TypeError):
        subject_condition.disjoin(literal_true)

    with pytest.raises(TypeError):
        subject_condition.implicate(literal_true)

    with pytest.raises(TypeError):
        subject_condition.delay(10)


def test_vehicle_type_condition():
    vehicle_type_condition = VehicleTypeCondition("passenger")

    passenger_vehicle_state = Mock()
    passenger_vehicle_state.vehicle_config_type = "passenger"

    truck_vehicle_state = Mock()
    truck_vehicle_state.vehicle_config_type = "truck"

    assert vehicle_type_condition.evaluate(vehicle_state=passenger_vehicle_state)
    assert not vehicle_type_condition.evaluate(vehicle_state=truck_vehicle_state)
