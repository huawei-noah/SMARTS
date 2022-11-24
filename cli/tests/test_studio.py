# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import os
import shutil
import tempfile
from xml.etree.ElementTree import ElementTree


def _compare_files(file1, file2):
    with open(file1) as f:
        items = [x.items() for x in ElementTree(file=f).iter()]

    with open(file2) as f:
        generated_items = [x.items() for x in ElementTree(file=f).iter()]

    sorted_items = sorted(items)
    sorted_generated_items = sorted(generated_items)
    if not sorted_items == sorted_generated_items:
        for a, b in zip(sorted_items, sorted_generated_items):
            assert a == b, f"{file1} is different than {file2}"


def test_scenario_generation_unchanged():
    number_of_comparisons_greater_than_0 = False
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            loc1 = temp_dir1 + "/scenarios"
            loc2 = temp_dir2 + "/scenarios"
            import re

            from cli.studio import build_scenarios

            _hashseed = os.getenv("PYTHONHASHSEED")
            assert _hashseed not in (None, "random"), f"PYTHONHASHSEED is {_hashseed}"

            shutil.copytree("scenarios/sumo", loc1)
            build_scenarios(True, True, [loc1], 42)

            shutil.copytree("scenarios/sumo", loc2)
            build_scenarios(True, True, [loc2], 42)

            for dirpath, dirnames, files in os.walk(loc1):
                if "traffic" in dirpath:
                    assert len(files) > 0
                    for file in files:
                        dir2 = re.sub(loc1, loc2, dirpath)
                        _compare_files(dirpath + "/" + file, dir2 + "/" + file)
                        number_of_comparisons_greater_than_0 = True

    assert number_of_comparisons_greater_than_0
