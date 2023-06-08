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
import pytest

from smarts.core.utils.type_operations import TypeSuite


class BaseA:
    pass


class BaseZ:
    pass


class C_Inherits_A(BaseA):
    pass


class D_Inherits_ZA(BaseZ, BaseA):
    pass


class E_Inherits_C(C_Inherits_A):
    pass


class F_Inherits_C(C_Inherits_A):
    pass


class UnrelatedBaseN:
    pass


def test_type_suite_insert():
    ts = TypeSuite(BaseA)

    ts.insert(BaseA())
    ts.insert(C_Inherits_A())
    ts.insert(D_Inherits_ZA())

    with pytest.raises(TypeError):
        ts.insert(BaseZ())  # pytype: disable=wrong-arg-types

    with pytest.raises(TypeError):
        ts.insert(UnrelatedBaseN())  # pytype: disable=wrong-arg-types

    a_expected = {
        BaseA,
        C_Inherits_A,
        D_Inherits_ZA,
    }
    a_derived = ts.get_all_by_type(BaseA)
    d_derived = ts.get_all_by_type(D_Inherits_ZA)
    assert {t.__class__ for t in a_derived} == a_expected
    assert {t.__class__ for t in ts.instances} == a_expected
    assert {t.__class__ for t in d_derived} == {D_Inherits_ZA}

    assert ts.get_by_type(E_Inherits_C) is None

    ts.insert(E_Inherits_C())
    ts.insert(F_Inherits_C())

    c_derived = ts.get_all_by_type(C_Inherits_A)
    assert len(c_derived) == 3
    assert {t.__class__ for t in c_derived} == {
        C_Inherits_A,
        E_Inherits_C,
        F_Inherits_C,
    }


def test_type_suite_remove():
    ts = TypeSuite(BaseA)

    inst = D_Inherits_ZA()
    ts.insert(BaseA())
    ts.insert(inst)
    ts.insert(E_Inherits_C())
    ts.remove(inst)

    assert len(ts.instances) == 2
    assert ts.get_by_id("D_Inherits_ZA") is None
    assert len(ts.get_all_by_type(D_Inherits_ZA)) == 0


def test_type_suite_remove_by_name():
    ts = TypeSuite(BaseA)

    inst = D_Inherits_ZA()
    ts.insert(inst)
    ts.remove_by_name(D_Inherits_ZA.__name__)

    assert len(ts.instances) == 0
    assert ts.get_by_id("D_Inherits_ZA") is None


def test_type_suite_remove_by_type():
    ts = TypeSuite(BaseA)

    inst = D_Inherits_ZA()
    ts.insert(inst)
    ts.remove_by_type(D_Inherits_ZA)

    assert len(ts.instances) == 0
    assert ts.get_by_id("D_Inherits_ZA") is None


def test_type_suite_clear_type():
    ts = TypeSuite(BaseA)

    types = [
        BaseA,
        C_Inherits_A,
        D_Inherits_ZA,
        E_Inherits_C,
        F_Inherits_C,
    ]

    for t in types:
        ts.insert(t())

    ts.clear_type(C_Inherits_A)

    a_expected = {
        BaseA,
        D_Inherits_ZA,
    }
    assert {t.__class__ for t in ts.get_all_by_type(BaseA)} == a_expected
    assert len(ts.get_all_by_type(C_Inherits_A)) == 0
