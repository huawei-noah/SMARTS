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
from unittest.mock import Mock

import pytest

from smarts.sstudio.sstypes import (
    CompoundCondition,
    Condition,
    ConditionOperator,
    ConditionState,
    ConditionTrigger,
    DependeeActorCondition,
    ExpireTrigger,
    LiteralCondition,
    NegatedCondition,
    OffRoadCondition,
    SubjectCondition,
    TimeWindowCondition,
    VehicleSpeedCondition,
    VehicleTypeCondition,
)

literal_true = LiteralCondition(ConditionState.TRUE)
literal_false = LiteralCondition(ConditionState.FALSE)
literal_before = LiteralCondition(ConditionState.BEFORE)
literal_expired = LiteralCondition(ConditionState.EXPIRED)


def test_condition_state():
    assert bool(ConditionState.TRUE)
    assert not bool(ConditionState.EXPIRED)
    assert not bool(ConditionState.BEFORE)
    assert not bool(ConditionState.FALSE)

    assert ConditionState.TRUE
    assert (not ConditionState.TRUE) == False
    assert (
        ConditionState.FALSE | ConditionState.BEFORE | ConditionState.EXPIRED
    ) in ~ConditionState.TRUE

    assert ConditionState.FALSE == False
    assert not ConditionState.FALSE
    assert (
        ConditionState.TRUE | ConditionState.BEFORE | ConditionState.EXPIRED
    ) in ~ConditionState.FALSE

    assert bool(ConditionState.EXPIRED) == False
    assert not ConditionState.EXPIRED
    assert (
        ConditionState.TRUE | ConditionState.BEFORE | ConditionState.FALSE
    ) in ~ConditionState.EXPIRED

    assert bool(ConditionState.BEFORE) == False
    assert not ConditionState.BEFORE
    assert (
        ConditionState.TRUE | ConditionState.FALSE | ConditionState.EXPIRED
    ) in ~ConditionState.BEFORE

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
    condition = Condition()

    with pytest.raises(NotImplementedError):
        condition.evaluate(actor_info=None)

    with pytest.raises(TypeError):
        condition.negation()

    with pytest.raises(TypeError):
        condition.conjunction(literal_true)

    with pytest.raises(TypeError):
        condition.disjunction(literal_true)

    with pytest.raises(TypeError):
        condition.implication(literal_true)

    with pytest.raises(TypeError):
        condition.trigger(10)

    with pytest.raises(TypeError):
        condition.expire(10)


def test_compound_condition():

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.CONJUNCTION,
    ) == literal_true.conjunction(literal_false)
    assert literal_true.conjunction(literal_true).evaluate() == ConditionState.TRUE

    assert (
        literal_expired.conjunction(literal_expired).evaluate()
        == ConditionState.EXPIRED
    )
    assert (
        literal_expired.conjunction(literal_true).evaluate() == ConditionState.EXPIRED
    )
    assert (
        literal_expired.conjunction(literal_before).evaluate() == ConditionState.EXPIRED
    )
    assert (
        literal_expired.conjunction(literal_false).evaluate() == ConditionState.EXPIRED
    )

    assert literal_before.conjunction(literal_true).evaluate() == ConditionState.BEFORE
    assert (
        literal_before.conjunction(literal_before).evaluate() == ConditionState.BEFORE
    )
    assert literal_before.conjunction(literal_false).evaluate() == ConditionState.BEFORE

    assert literal_false.conjunction(literal_true).evaluate() == ConditionState.FALSE
    assert literal_false.conjunction(literal_false).evaluate() == ConditionState.FALSE

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.DISJUNCTION,
    ) == literal_true.disjunction(literal_false)
    assert literal_true.disjunction(literal_true).evaluate() == ConditionState.TRUE
    assert literal_true.disjunction(literal_false).evaluate() == ConditionState.TRUE
    assert literal_true.disjunction(literal_before).evaluate() == ConditionState.TRUE
    assert literal_true.disjunction(literal_expired).evaluate() == ConditionState.TRUE

    assert (
        literal_before.disjunction(literal_before).evaluate() == ConditionState.BEFORE
    )
    assert (
        literal_before.disjunction(literal_expired).evaluate() == ConditionState.BEFORE
    )
    assert literal_before.disjunction(literal_false).evaluate() == ConditionState.BEFORE

    assert (
        literal_expired.disjunction(literal_expired).evaluate()
        == ConditionState.EXPIRED
    )

    assert literal_expired.disjunction(literal_false).evaluate() == ConditionState.FALSE
    assert literal_false.disjunction(literal_false).evaluate() == ConditionState.FALSE

    assert CompoundCondition(
        first_condition=literal_true,
        second_condition=literal_false,
        operator=ConditionOperator.IMPLICATION,
    ) == literal_true.implication(literal_false)
    assert literal_true.implication(literal_true).evaluate() == ConditionState.TRUE

    assert literal_true.implication(literal_expired).evaluate() == ConditionState.FALSE
    assert literal_true.implication(literal_before).evaluate() == ConditionState.FALSE
    assert literal_true.implication(literal_false).evaluate() == ConditionState.FALSE

    assert literal_expired.implication(literal_true).evaluate() == ConditionState.TRUE
    assert (
        literal_expired.implication(literal_expired).evaluate() == ConditionState.TRUE
    )
    assert literal_expired.implication(literal_before).evaluate() == ConditionState.TRUE
    assert literal_expired.implication(literal_false).evaluate() == ConditionState.TRUE

    assert literal_before.implication(literal_true).evaluate() == ConditionState.TRUE
    assert literal_before.implication(literal_expired).evaluate() == ConditionState.TRUE
    assert literal_before.implication(literal_before).evaluate() == ConditionState.TRUE
    assert literal_before.implication(literal_false).evaluate() == ConditionState.TRUE

    assert literal_false.implication(literal_true).evaluate() == ConditionState.TRUE
    assert literal_false.implication(literal_expired).evaluate() == ConditionState.TRUE
    assert literal_false.implication(literal_false).evaluate() == ConditionState.TRUE
    assert literal_false.implication(literal_false).evaluate() == ConditionState.TRUE


def test_condition_trigger():
    short_delay = 4
    long_delay = 10
    first_time_window_true = 5
    window_condition = TimeWindowCondition(4, 10)
    delayed_condition = window_condition.trigger(long_delay, persistent=False)

    assert delayed_condition == ConditionTrigger(
        inner_condition=window_condition,
        delay_seconds=long_delay,
        persistent=False,
    )

    # before
    time = 2
    assert (
        not delayed_condition.evaluate(time=time)
    ) and not window_condition.evaluate(time=time)
    # first true
    time = first_time_window_true
    assert (not delayed_condition.evaluate(time=time)) and window_condition.evaluate(
        time=time
    )
    # delay not expired
    time = first_time_window_true + long_delay - 1
    assert (
        not delayed_condition.evaluate(time=time)
    ) and not window_condition.evaluate(time=time)
    # delay expired
    time = first_time_window_true + long_delay
    assert delayed_condition.evaluate(time=time) and not window_condition.evaluate(
        time=time
    )
    # delay expired
    time = first_time_window_true + long_delay + 1
    assert delayed_condition.evaluate(time=time) and not window_condition.evaluate(
        time=time
    )
    # delay not expired
    time = first_time_window_true + long_delay - 1
    assert not delayed_condition.evaluate(time=time)

    # Test persistent true
    delayed_condition = window_condition.trigger(short_delay, persistent=True)
    time = first_time_window_true
    assert not delayed_condition.evaluate(time=time)
    time = first_time_window_true + short_delay
    assert delayed_condition.evaluate(time=time)
    time = first_time_window_true + long_delay
    assert not delayed_condition.evaluate(time=time)


def test_expire_trigger():
    end_time = 10
    before = end_time - 1
    after = end_time + 1
    expire_trigger = literal_true.expire(end_time)

    assert expire_trigger == ExpireTrigger(literal_true, end_time)

    assert expire_trigger.evaluate(time=before)
    assert not expire_trigger.evaluate(time=end_time)
    assert not expire_trigger.evaluate(time=after)

    first_time = 3
    expire_trigger = literal_true.expire(end_time, relative=True)
    assert expire_trigger.evaluate(time=first_time)
    assert expire_trigger.evaluate(time=first_time + before)
    assert expire_trigger.evaluate(time=end_time)
    assert not expire_trigger.evaluate(time=first_time + end_time)
    assert not expire_trigger.evaluate(time=first_time + after)


def test_dependee_condition():
    dependee_condition = DependeeActorCondition("leader")

    assert dependee_condition.evaluate(actor_ids={"mr", "leader"})
    assert not dependee_condition.evaluate(actor_ids={"other", "vehicle"})


def test_literal_condition():
    literal_true = LiteralCondition(ConditionState.TRUE)
    literal_false = LiteralCondition(ConditionState.FALSE)

    assert literal_false.evaluate() == ConditionState.FALSE
    assert literal_true.evaluate() == ConditionState.TRUE


def test_negated_condition():
    assert literal_false.negation() == NegatedCondition(literal_false)
    assert literal_true.negation() == NegatedCondition(literal_true)

    assert literal_false.negation().evaluate()
    assert not literal_true.negation().evaluate()


def test_off_road_condition():
    off_road_condition = OffRoadCondition()

    current_actor_road_status = Mock()
    current_actor_road_status.off_road = False
    current_actor_road_status.road = "c-ew"
    assert (
        off_road_condition.evaluate(current_actor_road_status=current_actor_road_status)
        == ConditionState.FALSE
    )

    current_actor_road_status = Mock()
    current_actor_road_status.off_road = True
    current_actor_road_status.road = None
    assert (
        off_road_condition.evaluate(current_actor_road_status=current_actor_road_status)
        == ConditionState.TRUE
    )

    current_actor_road_status = Mock()
    current_actor_road_status.off_road = False
    current_actor_road_status.road = None
    assert (
        off_road_condition.evaluate(current_actor_road_status=current_actor_road_status)
        == ConditionState.BEFORE
    )


def test_time_window_condition():
    start = 4
    between = 8
    end = 10

    window_condition = TimeWindowCondition(start=start, end=end)

    assert not window_condition.evaluate(time=start - 1)
    assert window_condition.evaluate(time=start)
    assert window_condition.evaluate(time=between)
    assert not window_condition.evaluate(time=end)
    assert not window_condition.evaluate(time=end + 1)


def test_subject_condition():
    subject_condition = SubjectCondition()

    with pytest.raises(NotImplementedError):
        subject_condition.evaluate(vehicle_state=None)

    with pytest.raises(TypeError):
        subject_condition.negation()

    with pytest.raises(TypeError):
        subject_condition.conjunction(literal_true)

    with pytest.raises(TypeError):
        subject_condition.disjunction(literal_true)

    with pytest.raises(TypeError):
        subject_condition.implication(literal_true)

    with pytest.raises(TypeError):
        subject_condition.trigger(10)

    with pytest.raises(TypeError):
        subject_condition.expire(10)


def test_vehicle_speed_condition():
    low = 30
    between = 50
    high = 100
    vehicle_speed_condition = VehicleSpeedCondition(low, high)

    slow_vehicle_state = Mock()
    slow_vehicle_state.speed = low - 10

    between_vehicle_state = Mock()
    between_vehicle_state.speed = between

    fast_vehicle_state = Mock()
    fast_vehicle_state.speed = high + 50

    assert not vehicle_speed_condition.evaluate(current_actor_state=slow_vehicle_state)
    assert vehicle_speed_condition.evaluate(current_actor_state=between_vehicle_state)
    assert not vehicle_speed_condition.evaluate(current_actor_state=fast_vehicle_state)


def test_vehicle_type_condition():
    vehicle_type_condition = VehicleTypeCondition("passenger")

    passenger_vehicle_state = Mock()
    passenger_vehicle_state.vehicle_config_type = "passenger"

    truck_vehicle_state = Mock()
    truck_vehicle_state.vehicle_config_type = "truck"

    assert vehicle_type_condition.evaluate(current_actor_state=passenger_vehicle_state)
    assert not vehicle_type_condition.evaluate(current_actor_state=truck_vehicle_state)
