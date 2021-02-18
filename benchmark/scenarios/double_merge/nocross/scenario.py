# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, Route, Scenario

missions = [
    Mission(Route(begin=("gneE17", 0, 10), end=("gneE5", 1, 100))),
    Mission(Route(begin=("gneE22", 0, 10), end=("gneE5", 0, 100))),
    Mission(Route(begin=("gneE17", 0, 25), end=("gneE5", 0, 100))),
    Mission(Route(begin=("gneE22", 0, 25), end=("gneE5", 1, 100))),
]
gen_scenario(
    Scenario(ego_missions=missions),
    output_dir=Path(__file__).parent,
)
